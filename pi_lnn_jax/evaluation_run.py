"""Evaluation-run recorder：eval 生命週期尾段的唯一 owner。

What:
    一個有狀態的 recorder，一個 evaluation run 建一次。它擁有「蓋 source
    provenance → 依凍結慣例命名 metric artifact → 寫盤」這段 per-evaluated-unit
    的生命週期尾段，把 metric 計算委派給既有的 `evaluate_field_series` seam。

Why 這是一個 module 而不是每支腳本各抄一遍（ADR-0002 的邏輯完成）:
    ADR-0002 讓 `evaluate_field_series` 成為 metric artifact 的唯一**寫入**路徑，
    但環繞它的生命週期——組 `SourceProvenance`（producer / code revision / dirty /
    inputs，並把已解析的 evaluation protocol 內嵌進 `details`）、依 granularity
    凍結的檔名命名、落盤——留給了 7 支 producer 各手寫一份。後果是具體的：

      - 某支漏蓋 `evaluation_protocol` 或 code revision，就會產出一份帶靜默
        provenance 洞的 artifact，而那份印記會跟著數字進論文表格，沒有測試攔得到。
      - 檔名慣例（`{stem}_re{Re}[_{method}][_m{modes}]` 等）沒有 owner，以格式
        字串散在每個 caller 裡。
      - `cost_accuracy` 已自刻了一個 local recorder（`_score`）——這個抽象是被
        需求證明過的，只是每支各刻各的。

Why protocol 必填:
    recorder 對 provenance 只給**一個**保證：每份 artifact 一定蓋上已解析的
    evaluation protocol。要讓這個保證成立，`protocol` 就必須是 `record()` 的必填
    參數、且其值由 recorder 寫入（不得由 `details_extras` 覆蓋）——這正是「靜默
    provenance 洞」的堵點（ADR-0003 讓 protocol 成為明示決定，本 module 讓它
    無法被漏蓋）。

Why 有狀態、一個 run 建一次:
    producer 身分、code revision、命名慣例是一個 run 的常量。建構時抓一次
    revision、per unit 呼 `record()`，於是這些常量只被陳述一次而不是每個
    evaluated unit 重寫（最強 locality）。

Why 命名不統一成單一模板:
    ADR-0002 依 granularity 凍結了檔名，且各 producer 的 granularity 是真實不同的
    （單-Re 不帶 `_re`、cost sweep 帶 `_m{modes}`、exp245 單筆為裸
    `metric_artifact.json`）。recorder 擁有命名 function，但由 caller 供有序後綴
    元件（`RunArtifactIdentity`）以逐字重現各自的凍結名；強行統一會改變輸出檔名。

recorder **不**擁有的（各 producer 保留）:
    重建（各 producer signature 迥異，無法統一）、protocol resolution 與對齊
    （既有 evaluation-protocol seam）、summary row（欄位各異）、compatibility
    projection 的 dict 與其寫入。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pi_lnn_jax.evaluation_protocol import EvaluationProtocol
from pi_lnn_jax.metric_artifact import (
    EvaluationContext,
    SourceProvenance,
    evaluate_field_series,
    freeze_provenance_details,
    repository_revision,
    write_metric_artifact,
)

#: recorder module 自己所在的 repo root（`pi_lnn_jax/evaluation_run.py` → 上兩層）。
#: 由 recorder 而非 caller 決定「哪個 revision」——這正是要收斂的重複：遷移前
#: 每支 producer 各自 `repository_revision(_REPO_ROOT)` 一遍。
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: `details` 裡由 recorder 保證寫入的鍵。它是 recorder 的獨佔欄位，caller 的
#: `details_extras` 不得帶它——否則「protocol 一定入 provenance」的保證就會被
#: 一份 caller 提供的值靜默覆蓋。
_RESERVED_DETAIL_KEY = "evaluation_protocol"


@dataclass(frozen=True)
class RunArtifactIdentity:
    """一份 metric artifact 的凍結檔名由哪些有序後綴元件構成。

    三個欄位對應三個真實的 granularity 軸；缺席的欄位就不出現在檔名裡，於是
    同一個 recorder 能逐字重現各 producer 不同的凍結名：

      - `re` + `method`                → `{stem}_re{Re}_{method}`
      - `re` + `method` + `modes`      → `{stem}_re{Re}_{method}_m{modes}`
      - `method`（單-Re，不帶 `_re`）  → `{stem}_{method}`
      - `re`                           → `{stem}_re{Re}`
      - 三者皆缺（單筆）               → `metric_artifact.json`（裸名，無 stem）

    命名的組裝在 `EvaluationRunRecorder`（命名慣例的 owner）；本型別只承載元件。
    """

    re: float | int | None = None
    method: str | None = None
    modes: int | None = None

    def __post_init__(self) -> None:
        # fail-fast：型別錯的元件若放行，只會在組檔名時炸在遠處，或更糟——
        # 悄悄產出一個歪掉的檔名。
        if self.re is not None and not isinstance(self.re, (int, float)):
            raise ValueError(f"identity.re 須為數值或 None，收到 {type(self.re).__name__}")
        if self.method is not None and (
            not isinstance(self.method, str) or not self.method.strip()
        ):
            raise ValueError("identity.method 須為非空字串或 None")
        if self.modes is not None and not isinstance(self.modes, int):
            raise ValueError(f"identity.modes 須為 int 或 None，收到 {type(self.modes).__name__}")


class EvaluationRunRecorder:
    """一個 evaluation run 的 stamp/write 生命週期尾段。

    一個 run 建一次 `EvaluationRunRecorder(producer, out_path)`：建構時抓一次
    code revision 與 dirty flag；per evaluated unit 呼 `record(...) -> projection`。
    """

    def __init__(self, producer: str, out_path: str | Path) -> None:
        if not str(producer).strip():
            raise ValueError("recorder 需要非空的 producer 身分")
        self._producer = str(producer)
        self._out_path = Path(out_path)
        # 建構時抓一次——一個 run 的 revision 是常量，不必每個 unit 重問 git。
        self._revision, self._code_dirty = repository_revision(_REPO_ROOT)

    @property
    def producer(self) -> str:
        return self._producer

    @property
    def out_path(self) -> Path:
        return self._out_path

    @property
    def code_revision(self) -> str:
        return self._revision

    @property
    def code_dirty(self) -> bool:
        return self._code_dirty

    def _artifact_filename(self, identity: RunArtifactIdentity) -> str:
        # 逐字重現 ADR-0002 凍結的檔名。元件順序固定為 re → method → modes；
        # 缺席即不出現，故省略 re 自然得到無 `_re` 的名。
        parts: list[str] = []
        if identity.re is not None:
            parts.append(f"re{identity.re:g}")
        if identity.method is not None:
            parts.append(identity.method)
        if identity.modes is not None:
            parts.append(f"m{identity.modes}")
        if not parts:
            # 單筆：exp245 那條的凍結名是裸 `metric_artifact.json`（無 stem）。
            return "metric_artifact.json"
        suffix = "".join(f"_{p}" for p in parts)
        return f"{self._out_path.stem}{suffix}.metric_artifact.json"

    def record(
        self,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        u_ref: np.ndarray,
        v_ref: np.ndarray,
        *,
        context: EvaluationContext,
        protocol: EvaluationProtocol,
        identity: RunArtifactIdentity,
        inputs: Sequence[tuple[str, str]],
        details_extras: Mapping[str, Any],
        run_measurements: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """評估一組重建、蓋 provenance、命名、落盤，回傳 compatibility projection。

        Args:
            u_pred, v_pred, u_ref, v_ref: 重建與參考速度場序列 `[T, Nx, Ny]`。
            context: 供 `evaluate_field_series` 的 evaluation context（含 times /
                grid_shape / viscosity 等）。由 caller 依協定載入時產生。
            protocol: **必填**。已解析的 evaluation protocol；recorder 保證把
                `protocol.to_provenance()` 寫進 provenance 的 `details`
                （鍵 `evaluation_protocol`）。這道保證即「靜默 provenance 洞」的堵點。
            identity: 有序後綴元件，重現此 producer 的凍結檔名。
            inputs: source provenance 的 inputs（sensor / dns / config 等來源對）。
            details_extras: 此 producer 特有的 provenance 細節（method、re_value、
                超參…）。**不得**含 `evaluation_protocol`——那是 recorder 的獨佔鍵。
            run_measurements: run-level cost measurement（如
                `{"recon_s_per_field": …}` / `{"wall_s": …}`），原樣透傳給寫入
                seam，以 cost measurement 語意落盤（ADR-0002），不塞進 provenance。

        Returns:
            `evaluate_field_series` 回傳的 compatibility projection（dict）。caller
            用它組自己那份欄位各異的 summary row。

        Raises:
            ValueError: 未給 `protocol`（或給了非 `EvaluationProtocol`）、未給
                `inputs`，或 `details_extras` 帶了保留鍵 `evaluation_protocol`。
        """
        if not isinstance(protocol, EvaluationProtocol):
            raise ValueError(
                "record() 需要已解析的 EvaluationProtocol——recorder 對每份 artifact "
                "只給一個保證：evaluation protocol 一定入 provenance；漏傳它等於放掉"
                "那個保證。"
            )
        if inputs is None:
            raise ValueError("record() 需要 inputs（source provenance 的來源對），不得為 None")
        if not isinstance(identity, RunArtifactIdentity):
            raise TypeError("record() 的 identity 須為 RunArtifactIdentity")
        if _RESERVED_DETAIL_KEY in details_extras:
            raise ValueError(
                f"details_extras 不得含 {_RESERVED_DETAIL_KEY!r}：它由 recorder 保證"
                "寫入，若由 caller 提供就會靜默覆蓋掉那個保證。"
            )

        # recorder 的 evaluation_protocol 放在最後，即使上面的守衛被繞過也仍勝出。
        details = freeze_provenance_details(
            {**dict(details_extras), _RESERVED_DETAIL_KEY: protocol.to_provenance()}
        )
        provenance = SourceProvenance(
            producer=self._producer,
            code_revision=self._revision,
            code_dirty=self._code_dirty,
            inputs=tuple(inputs),
            details=details,
        )
        artifact, projection = evaluate_field_series(
            u_pred, v_pred, u_ref, v_ref,
            context=context,
            provenance=provenance,
            run_measurements=run_measurements,
        )
        write_metric_artifact(
            self._out_path.with_name(self._artifact_filename(identity)), artifact
        )
        return projection

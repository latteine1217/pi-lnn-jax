"""Evaluation protocol：eval 的時間軸取樣與訓練 cadence 的關係。

What:
    一個 evaluation protocol 說三件事——sensor 以什麼 stride 取樣、DNS 以什麼
    stride 取樣、以及兩條時間軸要不要對齊。它由訓練 config 與呼叫端的明示模式
    共同決定，並隨產物落進 provenance。

Why 這是一個 module 而不是幾個 CLI 預設值:
    這個關係先前散在三處互不相容的地方：`scripts/slurm/submit_crossre.sh` 用 sed
    從 TOML 撈 `time_strides`（撈不到就拒絕提交）、`eval_exp301_enstrophy.sbatch`
    靠一句註解提醒要對齊、`eval_dropout_sweep.sbatch.tmpl` 明寫「與訓練
    time_strides 無關」。三種說法都對——它們描述的是三種**不同的協定**，而沒有
    任何地方把「這次是哪一種」寫成程式碼。

    後果是可量的：2026-08-06 實測 27 份 T20 系 `final_eval`，**12 份**的評估
    stride 與訓練宣告不符（多數訓練 8 / 評估 2，一份評估 1）。`evaluate_exp245`
    的 allclose 守衛攔不到——sensor 與 DNS 一起用錯的 stride 時兩者仍互相一致。

Why 模式必填:
    有預設就會有人吃到錯的那一個，而錯的那一個不會 crash，只會給出在別的時間
    解析度上算出來的數字。三種模式都是一等公民，呼叫端必須說出這次是哪一種。

Why 不統一成一組數值:
    正確的 stride 不是全域常數。82 份 config 的 `time_strides` 有 7 種取值
    （`[1]`×50、`[8]`×24、`[2]`×18、`[4]`、`[20]`、`[4,4,4,4,4]`、`[2,4]`），
    取決於該 sensor set 的原生 cadence。統一的是**來源**，不是數值。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

#: 訓練端在 `time_strides` 為空時採用的 stride。
#: 來源：`pipeline/kolmogorov/assembly.py` 的 `stride_arg = time_strides if time_strides else 2`
#: 與 `single_stride = int(time_strides[0]) if time_strides else 2`。
#: 這裡刻意複製而非自訂——若兩邊各有預設，「沒寫」的情況就會悄悄分岔。
TRAINING_DEFAULT_TIME_STRIDE = 2


class ProtocolMode(str, Enum):
    """三種都真實存在，且都有實驗依賴它。"""

    #: eval cadence 跟隨訓練 `time_strides`。sensor 是同一條軌跡的不同採樣率，
    #: 評估時必須跟上，否則 temporal encoder 收到訓練時沒見過的 context 長度。
    FOLLOW_TRAINING = "follow_training"

    #: eval 格點固定，與訓練 cadence 無關。snapshot-density 全族
    #: （`exp_snap_st{1,2,4,8,20}_*`）把 `time_strides` 當**自變數**掃 1→20；
    #: 若 eval 跟著變，st20 那臂會用比 st1 粗 20 倍的格點評，跨臂不可比。
    FIXED_GRID = "fixed_grid"

    #: sensor 時間軸刻意不等距且與 DNS 格點無關（thesis §7.2 的間歇評估）。
    #: query 走完整 DNS 格點、sensor context 走自己的序列，不做對齊檢查。
    #: 這是不同的協定，不是放寬對齊。
    SENSOR_TIME_INDEPENDENT = "sensor_time_independent"


@dataclass(frozen=True)
class EvaluationProtocol:
    """一次評估的時間軸取樣契約。不可變；原樣進 provenance。"""

    mode: ProtocolMode
    sensor_time_stride: int
    dns_time_stride: int
    #: 這次偏離（或跟隨）的對象。即使是 fixed_grid 也記下來——沒有它就分不出
    #: 「刻意固定」與「忘了帶旗標」。
    training_time_stride: int
    #: 這個 stride 為什麼是這個值，寫給讀產物的人看。
    basis: str
    sensor_T: int | None = None

    @property
    def aligns_sensor_to_dns(self) -> bool:
        return self.mode is not ProtocolMode.SENSOR_TIME_INDEPENDENT

    def to_provenance(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["mode"] = self.mode.value
        return d


def resolve_protocol(
    *,
    mode: ProtocolMode | str,
    training_time_strides: Sequence[int],
    re_index: int = 0,
    cli_time_stride: int | None = None,
    sensor_T: int | None = None,
    reason: str | None = None,
) -> EvaluationProtocol:
    """把「訓練宣告 + 呼叫端明示的模式」解析成一份協定。

    Args:
        mode: 三種之一，**必填**。字串會被驗證，打錯不會變成一個新模式。
        training_time_strides: 訓練 config 的 `data_kwargs.time_strides`（可為空）。
        re_index: multi-Re 時取第幾個 Re 的 stride。越界即 raise——靜默取 [0]
            會讓每個 Re 都用第一個 stride。
        cli_time_stride: 呼叫端明示的 stride。
        sensor_T: sensor 截斷幀數；None 表示用全部。
        reason: `fixed_grid` 必填，說明為何刻意偏離訓練 cadence。

    Raises:
        ValueError: 模式不合法、stride 非正、follow_training 下 CLI 與訓練不一致、
            fixed_grid 缺 stride 或缺 reason。
        IndexError: `re_index` 越界。
    """
    mode = ProtocolMode(mode)  # 打錯字不該變成一個新協定

    for name, value in (("cli_time_stride", cli_time_stride), ("sensor_T", sensor_T)):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} 須為正整數，收到 {value}")

    strides = [int(s) for s in training_time_strides]
    if any(s <= 0 for s in strides):
        raise ValueError(f"training_time_strides 含非正值：{strides}")
    if strides:
        if not 0 <= re_index < len(strides):
            raise IndexError(
                f"re_index={re_index} 超出 time_strides 長度 {len(strides)}；"
                "不靜默取 [0]——那會讓每個 Re 都用第一個 stride")
        training_stride = strides[re_index]
        training_basis = f"訓練 time_strides[{re_index}]={training_stride}"
    else:
        training_stride = TRAINING_DEFAULT_TIME_STRIDE
        training_basis = f"訓練 time_strides 未設 → fallback {training_stride}"

    if mode is ProtocolMode.FOLLOW_TRAINING:
        if cli_time_stride is not None and int(cli_time_stride) != training_stride:
            raise ValueError(
                f"follow_training：CLI stride {cli_time_stride} 與訓練宣告 "
                f"{training_stride} 不一致。兩者一起用錯的值時 sensor 與 DNS 仍會"
                f"互相對齊，內建檢查攔不到——故此處拒絕，不自行選一個。"
                f"（{training_basis}；若這次刻意要用不同格點，請改用 "
                f"mode=fixed_grid 並說明理由）")
        return EvaluationProtocol(
            mode=mode, sensor_time_stride=training_stride,
            # 對齊模式讓 DNS 保持完整候選池：先 stride 再匹配，可能把該匹配的幀剔掉。
            dns_time_stride=1,
            training_time_stride=training_stride,
            basis=training_basis, sensor_T=sensor_T,
        )

    if mode is ProtocolMode.FIXED_GRID:
        if cli_time_stride is None:
            raise ValueError(
                "fixed_grid 需明示 stride——固定格點的重點就是那個值由呼叫端決定")
        if not (reason or "").strip():
            raise ValueError(
                "fixed_grid 需說明 reason：刻意偏離訓練 cadence 是合法的"
                "（snapshot-density 全族靠它），但沉默的偏離與「忘了帶旗標」"
                "在產物上長得一模一樣")
        return EvaluationProtocol(
            mode=mode, sensor_time_stride=int(cli_time_stride), dns_time_stride=1,
            training_time_stride=training_stride,
            basis=f"固定格點 stride={int(cli_time_stride)}（{training_basis}）：{reason.strip()}",
            sensor_T=sensor_T,
        )

    # SENSOR_TIME_INDEPENDENT
    if cli_time_stride is None:
        raise ValueError(
            "sensor_time_independent 需明示 stride：它同時是 query 格點的解析度")
    stride = int(cli_time_stride)
    return EvaluationProtocol(
        mode=mode, sensor_time_stride=stride,
        # 此模式下 DNS 的 stride 是 query 解析度，不是對齊用的。
        dns_time_stride=stride,
        training_time_stride=training_stride,
        basis=(f"sensor 時間軸與 DNS 無關（thesis §7.2 間歇評估），"
               f"query 格點 stride={stride}（{training_basis}）"),
        sensor_T=sensor_T,
    )


@dataclass(frozen=True)
class AlignedEvaluation:
    """依協定載入並對齊後的評估輸入，外加它自己的 evaluation context。

    欄位是 `baseline_eval.load_aligned_re` 回傳面的超集，故既有呼叫端可逐鍵搬移。
    `context` 讓呼叫端不必各自拼一份——遷移前 `EvaluationContext` 有 7 個手動
    建構點，每一處都自行決定 times 從哪來。
    """

    protocol: EvaluationProtocol
    #: 反正規化回物理單位（classical baseline 比對 raw DNS 用）
    sensor_phys: Any
    #: 原樣的 normalized 值與其統計（模型輸入用；decoder 的輸入慣例）
    sensor_vals_normalized: Any
    norm_stats: Any
    sensor_pos: Any
    sensor_time: Any
    dns_u_eval: Any
    dns_v_eval: Any
    dns_t_eval: Any
    #: 壓力場；`with_pressure=False`（預設）時為 None
    dns_p_eval: Any
    dns_u_full: Any
    dns_v_full: Any
    eval_idx: list[int] | None
    T: int
    K: int
    N: int
    s: int
    Nprime: int
    context: "Any"


def load_for_evaluation(
    sensor_json,
    dns_path,
    *,
    protocol: EvaluationProtocol,
    case: str = "kolmogorov",
    viscosity: float | None = None,
    domain_length: float = 1.0,
    periodic: bool = True,
    grid_stride: int = 1,
    max_grid: int = 0,
    with_pressure: bool = False,
) -> AlignedEvaluation:
    """依協定載入 sensor + DNS 並對齊，回傳評估輸入與 evaluation context。

    對齊只有一份實作：`baseline_eval.match_sensor_dns_times` 的逐幀值匹配
    （gap 超容差或重複映射即 fail-fast）。遷移前有三份——兩份用 `allclose`
    比對導出的整數次採樣，在 sensor 時間不等距時直接失敗；值匹配是其中最嚴、
    也最通用的一份，且在另兩份會通過的情況下必然選中同一批幀。

    `sensor_time_independent` 模式不做對齊：query 走 DNS 自己的格點。
    """
    import numpy as np

    from pi_lnn_jax.baseline_eval import (
        choose_grid_stride,
        denormalize_sensors,
        match_sensor_dns_times,
    )
    from pi_lnn_jax.data import load_dns_from_path, load_sensors_from_path
    from pi_lnn_jax.metric_artifact import EvaluationContext

    d = load_sensors_from_path(sensor_json, time_stride=protocol.sensor_time_stride)
    n_sensor = d["sensor_vals"].shape[0]
    T = min(protocol.sensor_T, n_sensor) if protocol.sensor_T else n_sensor
    sensor_pos = np.asarray(d["sensor_pos"])
    sensor_time = np.asarray(d["sensor_time"][:T])
    sensor_phys = denormalize_sensors(d["sensor_vals"][:T], d["norm_stats"])

    if with_pressure:
        dns_u_full, dns_v_full, dns_p_full, dns_t_full = load_dns_from_path(
            dns_path, time_stride=protocol.dns_time_stride, return_p=True)
    else:
        dns_u_full, dns_v_full, dns_t_full = load_dns_from_path(
            dns_path, time_stride=protocol.dns_time_stride)
        dns_p_full = None
    N = int(dns_u_full.shape[1])
    s = choose_grid_stride(N, grid_stride=grid_stride, max_grid=max_grid)

    if protocol.aligns_sensor_to_dns:
        eval_idx = match_sensor_dns_times(sensor_time, dns_t_full)
        dns_u = np.asarray(dns_u_full[eval_idx][:, ::s, ::s])
        dns_v = np.asarray(dns_v_full[eval_idx][:, ::s, ::s])
        dns_t = np.asarray(dns_t_full)[eval_idx]
        dns_p = np.asarray(dns_p_full[eval_idx][:, ::s, ::s]) if dns_p_full is not None else None
    else:
        # query 走完整（已依 dns_time_stride 取樣的）DNS 格點；sensor 走自己的序列。
        eval_idx = None
        dns_u = np.asarray(dns_u_full[:, ::s, ::s])
        dns_v = np.asarray(dns_v_full[:, ::s, ::s])
        dns_t = np.asarray(dns_t_full)
        dns_p = np.asarray(dns_p_full[:, ::s, ::s]) if dns_p_full is not None else None

    context = EvaluationContext(
        case=case,
        times=tuple(float(t) for t in dns_t),
        periodic=periodic,
        domain_length=domain_length,
        viscosity=viscosity,
        pressure_evaluated=bool(with_pressure),
        grid_shape=(int(dns_u.shape[1]), int(dns_u.shape[2])),
    )
    return AlignedEvaluation(
        protocol=protocol, sensor_phys=sensor_phys,
        sensor_vals_normalized=np.asarray(d["sensor_vals"][:T]),
        norm_stats=d["norm_stats"], sensor_pos=sensor_pos,
        sensor_time=sensor_time, dns_u_eval=dns_u, dns_v_eval=dns_v, dns_t_eval=dns_t,
        dns_p_eval=dns_p,
        dns_u_full=dns_u_full, dns_v_full=dns_v_full, eval_idx=eval_idx,
        T=int(T), K=int(sensor_pos.shape[0]), N=N, s=int(s),
        Nprime=int(dns_u.shape[1]), context=context,
    )


def training_time_strides_from_config(config_path) -> list[int]:
    """從 TOML 讀 `data_kwargs.time_strides`（空 list 表示訓練端吃 fallback）。

    Why 在這裡而不是 `data.resolve_re_inputs`：後者回傳 `(Re, sensor, dns)` 三件組、
    刻意不帶協定資訊，且有 4 個呼叫端。與其改它的回傳型別，不如讓「協定從哪讀」
    也歸協定 module——這樣「eval 的 stride 從哪來」只有一個答案。
    """
    from pi_lnn_jax.config import load_config

    dk = load_config(config_path)["data_kwargs"]
    return [int(s) for s in dk.get("time_strides", [])]

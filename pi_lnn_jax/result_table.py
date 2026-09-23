"""Result table：把多份 producer projection 讀成統一的 (method × Re × split) 表。

What:
    一個 deep 讀側 module，把各 producer 寫出的 compatibility projection（那些
    `rows=[…]` 的多列 JSON，以及單筆 `metrics.json`）讀進一張以
    (method, Reynolds, split) 為鍵的 metric summary 表。它擁有三件事：

      1. **載入與正規化**：把 Shape A（多列 projection，metric 攤平在 row 上）與
         Shape B（單筆 metrics.json，metric 收在 `metrics_mean` 子塊）正規化成同一種
         evaluated unit，鍵為 (method, Re, split)。
      2. **method taxonomy（唯一一處）**：method id + 顯示 label，以及「一份 projection
         的 row 沒有 `method` 欄 → 用該 producer 宣告的 method id（例如 pi-con）」的
         身分規則。這些先前散在 `compare_baselines` 硬編碼／啟發式裡。
      3. **語意檢查過的 metric 讀取**：讀一個 metric value 一律經 metric_artifact 的
         typed 讀取路徑（`read_metric_summary` / ke_t_mape 雙語意規則），於是投影鍵被
         改名／漏掉／漂移時**大聲失敗**，而不是靜默回 None。這就是重點——讀側拿到寫側
         已有的「沒有語意就沒有值」保證。

Why 這是一個 module 而不是每支 consumer 各刻一遍:
    `compare_baselines` 手刻了 `_rows()`（硬挑 `d["rows"]` + 固定鍵）、硬編 method
    taxonomy `_METHODS`/`_LABELS`、「無 method 欄 → pi-con」啟發式、以及 (Re×method)
    的 `cell()` 視圖；`aggregate_pv_campaign` 則自己 `d["metrics_mean"].get(k, nan)`
    （漏鍵靜默變 NaN + warning）。兩者都各自把「投影鍵長什麼樣」焊死在腳本裡——
    投影鍵一旦漂移，consumer 只會安靜給錯數字。這個 module 把讀取收斂成一個 seam，
    metric 存取一律走 metric_artifact 的語意層（storage key 是那邊的實作細節）。

Why 從 metric_artifact 的定義派生、不手抄:
    讀側 taxonomy（definition id ↔ storage key）**派生自** `metric_artifact` 自己的
    `_METRIC_DEFINITIONS`，不在本檔手抄一份平行表。手抄會漂移；派生不會。這面
    鏡像寫側：ADR-0002 讓寫入單源，本 module 讓讀取也單源。

本 module **不**擁有的（留在 caller）:
    統計量（mean / sd / Welch 留在 campaign aggregator）、繪圖、以及 compatibility
    projection 的**寫入**（那是 recorder / 寫側，見 `evaluation_run.py` 與 ADR-0004）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence

from pi_lnn_jax.metric_artifact import (
    KE_T_MAPE_POINTWISE_V2,
    KE_T_MAPE_SPATIALMEAN_V1,
    METRIC_ARTIFACT_SCHEMA_VERSION,
    VORTICITY_REL_ERR_TIME_MEAN_V1,
    LegacyInterpretation,
    MetricUnavailable,
    read_metric_summary,
    _METRIC_DEFINITIONS,
)

__all__ = [
    "Split",
    "EvaluatedUnit",
    "ResultTable",
    "read_metric_value",
    "metric_definition_id",
    "method_label",
    "KE_T_MAPE_SPATIALMEAN_V1",
    "KE_T_MAPE_POINTWISE_V2",
    "VORTICITY_REL_ERR_TIME_MEAN_V1",
]


# ── metric definition ↔ storage key（派生自 write-side schema，唯一來源）─────────
#: definition id → 該量在 legacy projection 的 `metrics_mean` 儲存鍵。派生自
#: `_METRIC_DEFINITIONS`，故不可能與寫側漂移。KE-MAPE 家族刻意不從這裡讀（見下）。
_STORAGE_KEY_BY_DEFINITION: dict[str, str] = {
    definition.definition_id: key for key, definition in _METRIC_DEFINITIONS.items()
}
#: `read_metric_summary` 用的 vorticity **summary** id（time-mean）也對到同一個 storage
#: 鍵；補這個別名，讓 caller 用哪個 vorticity 常量都讀得到（兩者皆走嚴格讀，避免
#: read_metric_summary 對 null 值 `float(None)` 的崩潰）。
_STORAGE_KEY_BY_DEFINITION.setdefault(VORTICITY_REL_ERR_TIME_MEAN_V1, "omega_rel_err")

#: storage key → definition id。給仍以投影鍵字串命名的 caller（如 `--metric u_rel_err`）
#: 一座過橋：把 storage 細節解析成語意名。
_DEFINITION_ID_BY_STORAGE_KEY: dict[str, str] = {
    key: definition.definition_id for key, definition in _METRIC_DEFINITIONS.items()
}

#: projection row 上「哪些欄是 metric」= 寫側註冊過的 storage 鍵。正規化 row 時用它把
#: metric 欄與 identity/aux 欄（Re / method / held_out / eval_grid / K / wall_s …）分開。
_KNOWN_STORAGE_KEYS: frozenset[str] = frozenset(_METRIC_DEFINITIONS)

#: 一定要走 `read_metric_summary` 的 definition —— ke_t_mape 的 pointwise / spatialmean
#: 雙語意陷阱（scripts/CLAUDE.md §6）就在這裡，storage 鍵在 `ke_t_errors` 而非
#: `metrics_mean`，且需要語意標記或明示 legacy interpretation 才讀得對。
_KE_MAPE_FAMILY: frozenset[str] = frozenset({KE_T_MAPE_SPATIALMEAN_V1, KE_T_MAPE_POINTWISE_V2})


# ── method taxonomy（唯一一處）────────────────────────────────────────────────
#: 已知 method 的顯示 label 與**規範順序**。先前硬編在 `compare_baselines._LABELS`
#: 與 `_METHODS`。順序即論文四方表的欄序。
_METHOD_LABELS: dict[str, str] = {
    "interp_linear": "interp",
    "gappy_pod": "gappy-POD (per-Re)",
    "gappy_cross_re": "gappy-POD (cross-Re)",
    "shred": "SHRED",
    "pi-con": "PI-CON",
}
_METHOD_ORDER: tuple[str, ...] = tuple(_METHOD_LABELS)


def method_label(method_id: str) -> str:
    """method id → 顯示 label；未登錄的 method 退回其 id（不猜、不失敗）。"""
    return _METHOD_LABELS.get(method_id, method_id)


def metric_definition_id(projection_key: str) -> str:
    """projection/storage 鍵 → 語意 metric definition id（給以鍵命名的 caller 過橋）。

    Raises:
        KeyError: 該鍵不是寫側註冊過的 metric —— 未註冊的鍵沒有語意，不給讀。
    """
    try:
        return _DEFINITION_ID_BY_STORAGE_KEY[projection_key]
    except KeyError:
        raise KeyError(
            f"unregistered projection key {projection_key!r}: not a known metric definition"
        ) from None


class Split(str, Enum):
    """一個 evaluated unit 屬於訓練 Re 或留出 Re。由 `held_out` 布林導出。"""

    IN_TRAIN = "in_train"
    HELD_OUT = "held_out"


def split_from_held_out(held_out: bool) -> Split:
    return Split.HELD_OUT if bool(held_out) else Split.IN_TRAIN


def read_metric_value(
    payload: Mapping[str, Any],
    definition_id: str,
    *,
    legacy_interpretation: LegacyInterpretation | None = None,
) -> float | None:
    """從一份 projection payload 讀一個 metric value，語意檢查過。

    這是本 module 的深核心，兩個 consumer 都靠它拿「沒有語意就沒有值」的保證：

      - **KE-MAPE 家族**（pointwise / spatialmean）與 canonical artifact：一律委派
        `read_metric_summary`，於是 ke_t_mape 的雙語意、legacy interpretation、
        availability 都由那個唯一語意層裁決（不對就 raise，不靜默給錯的量）。
      - **其餘 registered metric**（legacy projection）：解析 definition → storage 鍵，
        **嚴格**讀 `metrics_mean`。鍵不在 → raise（投影鍵漂移就是在這裡被抓到）；
        值為 null / 非有限 → 回 None（那是 availability，不是漂移）。

    Args:
        payload: 一份 compatibility projection（`metrics.json` 形，含 `metrics_mean`；
            legacy artifact 另含 `ke_t_errors`），或一份 canonical metric artifact。
        definition_id: 要讀的 metric definition id（語意名，非 storage 鍵）。
        legacy_interpretation: 對未標記語意的 legacy artifact 讀 ke_t_mape 時的明示
            interpretation（原樣轉給 `read_metric_summary`）。

    Returns:
        float 值；量存在但不適用/不可得（null）時回 None。

    Raises:
        KeyError: `definition_id` 不是已知 metric definition。
        MetricUnavailable: projection 缺 `metrics_mean`，或該 metric 的儲存鍵不在
            projection 裡（投影鍵漂移／漏寫）。
        （ke_t_mape 語意不明時由 `read_metric_summary` 拋 AmbiguousMetricSemantics。）
    """
    if (
        payload.get("schema_version") == METRIC_ARTIFACT_SCHEMA_VERSION
        or definition_id in _KE_MAPE_FAMILY
    ):
        return read_metric_summary(
            payload, definition_id, legacy_interpretation=legacy_interpretation
        ).value

    storage_key = _STORAGE_KEY_BY_DEFINITION.get(definition_id)
    if storage_key is None:
        raise KeyError(f"unknown metric definition: {definition_id!r}")
    metrics_mean = payload.get("metrics_mean")
    if metrics_mean is None:
        raise MetricUnavailable("projection carries no metrics_mean summary block")
    if storage_key not in metrics_mean:
        raise MetricUnavailable(
            f"{definition_id} ({storage_key!r}) absent from projection metrics_mean "
            "— renamed/dropped projection key?"
        )
    raw = metrics_mean[storage_key]
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class EvaluatedUnit:
    """一次評估的表格單元：identity (method, Re, split) + 其 metric summary 證據。

    `payload` 已正規化成 `metrics.json` 形（頂層有 `metrics_mean`；來自單筆 metrics.json
    時另帶 `ke_t_errors`），故 `value()` 能一致地委派給 `read_metric_value`。
    """

    method: str
    reynolds: float
    split: Split
    payload: Mapping[str, Any]

    def value(
        self,
        definition_id: str,
        *,
        legacy_interpretation: LegacyInterpretation | None = None,
    ) -> float | None:
        """讀本單元的一個 metric value（語意檢查過；見 `read_metric_value`）。"""
        return read_metric_value(
            self.payload, definition_id, legacy_interpretation=legacy_interpretation
        )

    @property
    def label(self) -> str:
        return method_label(self.method)


def _unit_from_row(row: Mapping[str, Any], default_method: str | None) -> EvaluatedUnit:
    """把一份多列 projection 的一個 row 正規化成 EvaluatedUnit。

    「無 `method` 欄 → 用 producer 宣告的 default method」的身分規則就落在這：row 帶
    `method` 就用它，否則退到 `default_method`；兩者皆無就 raise（不像原本靜默標成
    "?" 而混進表裡）。
    """
    if "Re" not in row:
        raise KeyError("projection row lacks required identity 'Re'")
    if "held_out" not in row:
        raise KeyError("projection row lacks required identity 'held_out'")
    method = row.get("method") or default_method
    if not method:
        raise ValueError(
            "projection row has no 'method' column and the projection declared no "
            "default method — cannot assign a method identity"
        )
    # 只取寫側註冊過的 metric 欄；identity/aux 欄（eval_grid / K / wall_s / re_norm …）
    # 與 picon row 內嵌的 `metrics_mean` 子塊都排除在外。
    metrics_mean = {k: v for k, v in row.items() if k in _KNOWN_STORAGE_KEYS}
    return EvaluatedUnit(
        method=str(method),
        reynolds=float(row["Re"]),
        split=split_from_held_out(bool(row["held_out"])),
        payload={"metrics_mean": metrics_mean},
    )


class ResultTable:
    """以 (method, Re, split) 為鍵的 evaluated-unit 表。

    小介面、深實作：
      - 建構：`from_projection` / `from_projections`（吃各 producer 的 `rows` projection）。
      - 查詢：`cell` 取單元、`units` 依 method/split 疊代、`value` 直取一個 metric。
      - 盤點：`methods` / `reynolds_numbers` / `splits` / `reynolds_split`。
    """

    def __init__(self, units: Sequence[EvaluatedUnit]) -> None:
        index: dict[tuple[str, float, Split], EvaluatedUnit] = {}
        for unit in units:
            key = (unit.method, unit.reynolds, unit.split)
            if key in index:
                # 同一 (method, Re, split) 出現兩次＝重複載入或資料矛盾。大聲失敗，
                # 不靜默用後者蓋前者（那正是「安靜給錯數字」的溫床）。
                raise ValueError(
                    f"duplicate evaluated unit for method={unit.method!r} "
                    f"Re={unit.reynolds!r} split={unit.split.value!r}"
                )
            index[key] = unit
        self._units: tuple[EvaluatedUnit, ...] = tuple(units)
        self._index = index

    # ── 建構 ──────────────────────────────────────────────────────────────
    @classmethod
    def from_projection(
        cls, payload: Mapping[str, Any], *, default_method: str | None = None
    ) -> "ResultTable":
        """從一份多列 projection（含 `rows`）建表。"""
        return cls.from_projections([(payload, default_method)])

    @classmethod
    def from_projections(
        cls,
        projections: Iterable[Mapping[str, Any] | tuple[Mapping[str, Any], str | None]],
    ) -> "ResultTable":
        """合併多份多列 projection 建一張表。

        每個項目是 `payload` 或 `(payload, default_method)`。`default_method` 承載
        「無 method 欄 → 此 producer 的 method id」規則（如 picon → "pi-con"）。
        """
        units: list[EvaluatedUnit] = []
        for item in projections:
            payload, default_method = item if isinstance(item, tuple) else (item, None)
            if "rows" not in payload:
                raise KeyError("projection has no 'rows'; not a multi-unit projection")
            for row in payload["rows"]:
                units.append(_unit_from_row(row, default_method))
        return cls(units)

    # ── 盤點 ──────────────────────────────────────────────────────────────
    def methods(self) -> tuple[str, ...]:
        """在場的 method，依 taxonomy 規範順序排列（未登錄者殿後、按字母序）。"""
        present = {u.method for u in self._units}
        known = [m for m in _METHOD_ORDER if m in present]
        unknown = sorted(present - set(_METHOD_ORDER))
        return tuple(known + unknown)

    def reynolds_numbers(self) -> tuple[float, ...]:
        return tuple(sorted({u.reynolds for u in self._units}))

    def splits(self) -> tuple[Split, ...]:
        present = {u.split for u in self._units}
        return tuple(s for s in Split if s in present)

    def reynolds_split(self, reynolds: float) -> Split:
        """某個 Re 的 split。一個 Re 全域屬訓練或留出；跨 method 不一致就 raise。"""
        re = float(reynolds)
        found = {u.split for u in self._units if u.reynolds == re}
        if not found:
            raise KeyError(f"no evaluated unit at Re={reynolds!r}")
        if len(found) > 1:
            raise ValueError(
                f"Re={reynolds!r} appears in multiple splits {sorted(s.value for s in found)!r}"
            )
        return next(iter(found))

    # ── 查詢 ──────────────────────────────────────────────────────────────
    def cell(
        self, method: str, reynolds: float, *, split: Split | None = None
    ) -> EvaluatedUnit | None:
        """取 (method, Re) 單元；不在場回 None。

        `split` 省略時自動判定（一個 (method, Re) 實務上只有一個 split）；若同時存在
        多個 split 則 raise（要求 caller 指明），不靜默挑一個。
        """
        re = float(reynolds)
        if split is not None:
            return self._index.get((method, re, split))
        hits = [u for u in self._units if u.method == method and u.reynolds == re]
        if not hits:
            return None
        if len(hits) > 1:
            raise ValueError(
                f"method={method!r} Re={reynolds!r} spans multiple splits; pass split="
            )
        return hits[0]

    def units(
        self, *, method: str | None = None, split: Split | None = None
    ) -> tuple[EvaluatedUnit, ...]:
        """疊代單元，可依 method / split 過濾（服務 per-method/split 的均值分組）。"""
        return tuple(
            u
            for u in self._units
            if (method is None or u.method == method)
            and (split is None or u.split == split)
        )

    def value(
        self,
        method: str,
        reynolds: float,
        definition_id: str,
        *,
        split: Split | None = None,
        legacy_interpretation: LegacyInterpretation | None = None,
    ) -> float | None:
        """直取 (method, Re[, split]) 的一個 metric value。

        單元不在場 → None（空格）；單元在場但 metric 儲存鍵漂移 → `read_metric_value`
        raise（不靜默回 None）。這正是空格與漂移的分野。
        """
        unit = self.cell(method, reynolds, split=split)
        if unit is None:
            return None
        return unit.value(definition_id, legacy_interpretation=legacy_interpretation)

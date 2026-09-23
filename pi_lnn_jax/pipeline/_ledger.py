"""RNG/schedule ledger —— 診斷用 first-divergence oracle。

What: 逐步記錄一個 training step 的 host-side 決策（RNG 消費結果、課程排程值、
      取樣結果的 digest），供新舊實作比對、定位第一個 divergence。

Why: 「跑完 N 步比 params」只能告訴你壞了，不能告訴你哪一步壞的。重構訓練
     入口最大的風險是 RNG 消費時序被靜默改動（closure capture 改變、static
     value 被提前求值、某條 key 提早 split），這類錯誤一般單元測試抓不到，
     而且只會表現為「數字對不上」。

紀律（spec §6，不得放寬）：
  - 由環境變數 PILNN_RNG_LEDGER gate，預設關閉；關閉時 get_ledger() 回 None，
    呼叫端一律以 `if ledger is not None:` 包住，production 路徑零 overhead。
  - 只記錄既有計算結果，不得新增 RNG split、sampling 或陣列重算。
  - 大陣列只存 deterministic digest，不把整份陣列搬回 host / 寫進檔案。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

_ENV_VAR = "PILNN_RNG_LEDGER"

#: 尾列的 step 值。尾列不是某一步，是「這份錄製」的中繼資料
#: （params/opt digest、argv、config 路徑、資料衍生量、執行環境）。
TAIL_STEP = -1


def ledger_enabled() -> bool:
    """環境變數非空且非 '0' 時啟用。"""
    val = os.environ.get(_ENV_VAR, "")
    return bool(val) and val != "0"


def environment() -> dict:
    """這份錄製的執行環境——決定它的 digest 能不能拿來跟別份比。

    Why 必須記：digest 只在特定平台上有意義。同一份程式碼、同一組輸入，
    arm64 與 x86_64 給出不同的 `params_digest`（job 4740 對照本機實測：
    cylinder c1 得 b4549f5078513bbc vs 7ee61263e0cf0d5a）；GPU 更是同機
    兩跑就不同（job 4729）。沒有這幾個欄位，「digest 對不上」無從區分是
    程式碼變了還是機器換了——而那兩件事的處置完全相反。

    `backend` 尤其承重：它直接說明這份錄製能不能當位元判準，
    比從命令列旗標反推可靠。
    """
    import platform

    import jaxlib

    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
    }


def _hash_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    """把 shape / dtype / 位元組餵進 hasher。

    ascontiguousarray 是必要的：ledger 會收到切片（如 sensor_idx 的 view），
    非連續記憶體直接 tobytes 會取到 stride 之外的內容。
    """
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())


def digest(x: Any) -> str:
    """對單一 array-like 取 deterministic digest（sha256 前 16 hex）。"""
    h = hashlib.sha256()
    _hash_array(h, np.asarray(x))
    return h.hexdigest()[:16]


def params_digest(params: Any) -> str:
    """對 params pytree 取 deterministic digest。

    依 tree path 走訪，故不受 dict 插入順序影響；path 本身也進 hash，
    避免「兩個 leaf 互換位置」被誤判為相同。
    """
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    h = hashlib.sha256()
    for path, leaf in leaves:
        h.update(jax.tree_util.keystr(path).encode())
        _hash_array(h, np.asarray(leaf))
    return h.hexdigest()[:16]


def environment_mismatch(recorded: dict | None) -> str | None:
    """錄製環境與當前環境是否相容到可以比 digest；相容回 None，否則回不相容的理由。

    Why 回字串而非 bool：呼叫端要把理由印給人看。「不能比」而不說哪裡不同，
    使用者只能自己去猜是換了機器、換了 backend 還是升級了套件。

    `None`（尾列沒記環境，即 §8.3 第 7 項之前錄的 fixture）一律視為
    **無從判斷**而非相容——當成相容會讓跨平台比對在錯誤前提下靜靜進行。
    """
    if recorded is None:
        return "尾列未記錄執行環境，無從判斷 digest 是否可比（見 wave2 spec §8.3）"
    here = environment()
    for key in sorted(set(here) | set(recorded)):
        if recorded.get(key) != here.get(key):
            return (f"執行環境不符：{key} 錄製時為 {recorded.get(key)!r}，"
                    f"當前為 {here.get(key)!r}")
    return None


class Ledger:
    """逐步紀錄容器。純 host-side，不持有任何 device array。"""

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._notes: dict = {}

    def note(self, **fields: Any) -> None:
        """暫存尾列欄位。給「值在早期階段算出、尾列到最後才寫」的量用。

        Why 需要這個通道：`init_params_digest` 只有 `initialize` 手上有，
        而尾列由 `finalize` 寫；把它塞進 `TrainingState` 會讓「state 決定數值」
        這個讀法失效（它不參與任何數值決策，是純觀測量），塞進各案的 journal
        則兩案不對稱（cylinder 沒有 journal）。放在 ledger 內部最小且對稱。
        """
        self._notes.update(fields)

    def record(self, step: int, **fields: Any) -> None:
        """記一列。尾列（`TAIL_STEP`）自動帶上執行環境。

        Why 自動而非要求各 case 自己帶：尾列是 ledger 自己定義的概念，
        環境屬於「這份錄製」而不屬於任一案的邏輯。放在這裡，兩案不會漂移，
        將來第三個 case 也自動有。呼叫端明示 `environment=` 時不覆蓋
        （replay 端重建尾列做測試時需要自己指定）。
        """
        step = int(step)
        if step == TAIL_STEP:
            if "environment" not in fields:
                fields["environment"] = environment()
            # note() 暫存的欄位併入尾列；呼叫端明示者優先。
            fields = {**self._notes, **fields}
        self._rows.append({"step": step, **fields})

    @property
    def rows(self) -> list[dict]:
        return self._rows

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._rows, f, indent=2, sort_keys=True)


_LEDGER: Ledger | None = None


def get_ledger() -> Ledger | None:
    """啟用時回傳 process-wide Ledger，否則 None。

    呼叫端必須以 None 判斷跳過，不要在關閉狀態下建構任何診斷資料。
    """
    global _LEDGER
    if not ledger_enabled():
        return None
    if _LEDGER is None:
        _LEDGER = Ledger()
    return _LEDGER


def reset_ledger() -> None:
    """測試用：清掉 process-wide 實例。"""
    global _LEDGER
    _LEDGER = None

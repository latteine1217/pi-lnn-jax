"""Orbax-backed checkpoint manager for the JAX/Flax pi-lnn POC.

What: 將 Flax train state（params + opt_state + step + rng_key + 可選的
       GradNormState / ALState）以原子化、可保留歷史步數的方式存到磁碟，
       並支援 resume 還原。

Why: pi-lnn EXP-082 曾因 resume 後 silent state corruption（ScheduleFree
     internal step 與 GradNorm log_weights 未被一併還原）導致 KE rel-err
     98.5% 的 catastrophic collapse。本模組強制完整序列化決定優化軌跡的狀態
     （params、optax state 含 ScheduleFree step、log_weights、AL lambda、step），
     確保 resume 後優化軌跡續得上。

     resume 連續性的**跨行程**驗證不在本模組——它做不到。要比對「存檔前」與
     「還原後」的軌跡，必須同時握有兩個行程的狀態，而真實 resume 裡它們不同時存在。
     那件事由驗收層 B 承擔：`scripts/slurm/verify_ckpt_compat.sbatch.tmpl`
     讓兩側各自從同一份 ckpt 續跑再比對。本模組內曾有一個
     `verify_resume_continuity` 宣稱守這件事，但它要求呼叫端同時傳入 before/after
     兩個 state，因此結構上只能在測試裡用，零 production 呼叫者——已移除。

     注意（非 bit-deterministic）：僅存單一 rng_key（collocation 流）；multi-Re
     index RNG、continuous-Re RNG 與 RAR state 未存，resume 後這些隨機流是
     re-derived → 統計等價但非逐位元相同。RAR 候選池每次重抽，無學習狀態遺失，
     故不影響收斂；若需完全 bit-deterministic resume，須將上述流一併納入 TrainState。

設計重點：
  - 用 `orbax.checkpoint.CheckpointManager` + `PyTreeSave/Restore` 配
    `Composite` 包裝；PyTreeSave 容得下 NamedTuple 內部混雜的 jnp.ndarray
    與 Python `str/float` 靜態 metadata（StandardSave 會在 str 上炸）。
  - Restore 必須提供一個型別匹配的 template（實務上就是 reference TrainState）
    才能還原成原本的 NamedTuple 型別；否則 orbax 會把 NamedTuple 攤平成 dict，
    破壞下游程式碼的型別假設。
  - `save_interval_steps` 由 manager 決定真正落盤頻率；`force=True` 可
    bypass 該檢查（如訓練結束強制 dump final ckpt）。
  - 不寫 multi-host / GPU sharding 邏輯（POC scope = 單機）。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from .model_factory import fingerprint_diff, model_fingerprint


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TrainState
# ─────────────────────────────────────────────────────────────────────────────

class TrainState(NamedTuple):
    """Functional train state snapshot for checkpointing.

    所有欄位必須是 jax-traceable 或 pytree-friendly：
      - params:         Flax params pytree（dict of jnp.ndarray）
      - opt_state:      optax optimizer state pytree（含 ScheduleFree 內部
                        `_global_step` 若有；optax NamedTuple 會被正確還原）
      - step:           int32 scalar jnp.ndarray
      - rng_key:        jax PRNGKey (uint32[2])
      - gradnorm_state: GradNormState NamedTuple 或 None
      - al_state:       ALState NamedTuple 或 None
      - lra_state:      **永遠 None**。LRA controller 已於 2026-08-03 移除（實測與
                        GradNorm 數值等價、production 零使用，見架構審查候選 L）。
                        欄位保留純為 ckpt 格式相容——實測移除它會讓既有 ckpt 全部
                        restore 失敗（orbax 連 None 值也寫 metadata entry，樹結構不符）。
                        不要因為「它總是 None」就刪掉它。

    Migration（2026-06 brdr 移除）：brdr_state 欄位已移除。orbax restore 比對 tree
    structure，移除前存的 checkpoint（含 brdr_state leaf，即使值為 None 也會序列化成
    tree entry）與新 template 結構不符 → 無法用 production restore path 還原。
    pre-removal ckpt 需從 dict 手動遷移或重訓，不可直接 resume。

    Why NamedTuple 而非 dataclass：
      pi_lnn_jax.losses 已用 NamedTuple；維持 functional pure 風格、可被
      `jax.tree_util.tree_map` 自然遞迴而無須註冊。
    """
    params: Any
    opt_state: Any
    step: jnp.ndarray
    rng_key: jnp.ndarray
    gradnorm_state: Any = None
    al_state: Any = None
    lra_state: Any = None


# ─────────────────────────────────────────────────────────────────────────────
# CheckpointManager wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Thin wrapper around `orbax.checkpoint.CheckpointManager`.

    保存策略：所有 TrainState 欄位透過單一 PyTree item（item name='state'）
    一次寫入；恢復時必須傳入 abstract template 或一個 reference TrainState。

    Args:
        directory:           checkpoint 根目錄（會自動建立；建議使用絕對路徑）
        max_to_keep:         最多保留最新幾個 step；超出時自動刪除最舊
        save_interval_steps: 每 N 步才真正落盤一次；force=True 可繞過

    Note: 為 POC 場景禁用 async checkpointing（簡化測試；單機訓練 IO 量小，
          synchronous 寫不會明顯 block）。若未來需要 async，在 options 改
          `enable_async_checkpointing=True` 即可。
    """

    _ITEM_NAME: str = "state"

    def __init__(
        self,
        directory: str | Path,
        max_to_keep: int = 3,
        save_interval_steps: int = 100,
    ) -> None:
        if max_to_keep is not None and max_to_keep < 1:
            raise ValueError(f"max_to_keep 必須 ≥ 1，收到 {max_to_keep}")
        if save_interval_steps < 1:
            raise ValueError(f"save_interval_steps 必須 ≥ 1，收到 {save_interval_steps}")

        # orbax 要求 absolute path；用 Path.resolve() 統一避免「相對於 cwd」歧義
        self._directory: Path = Path(directory).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
            create=True,
            enable_async_checkpointing=False,  # POC 同步寫；保證測試確定性
            cleanup_tmp_directories=True,
        )
        self._mgr: ocp.CheckpointManager = ocp.CheckpointManager(
            self._directory,
            options=options,
            item_names=(self._ITEM_NAME,),
        )

    # ─── public API ──────────────────────────────────────────────────────

    def save(
        self,
        step: int,
        train_state: TrainState,
        force: bool = False,
    ) -> bool:
        """落盤一個 step 的 TrainState。

        Returns:
            True  若 orbax 實際寫入（含 save_interval_steps gating 通過）
            False 若被 gating 跳過（且 force=False）

        Note: orbax `save()` 本身回傳 bool；force=True 強制忽略 interval 檢查。
        """
        if not isinstance(train_state, TrainState):
            raise TypeError(
                f"train_state 必須是 ckpt.TrainState，收到 {type(train_state).__name__}"
            )
        wrote = self._mgr.save(
            int(step),
            args=ocp.args.Composite(**{self._ITEM_NAME: ocp.args.PyTreeSave(train_state)}),
            force=force,
        )
        if wrote:
            # wait_until_finished 在 sync mode 下基本是 no-op；保留以便日後改 async
            self._mgr.wait_until_finished()
            logger.debug("ckpt saved at step=%d → %s", step, self._directory)
        return bool(wrote)

    def restore(
        self,
        step: int | None = None,
        reference_state: TrainState | None = None,
    ) -> TrainState:
        """從 ckpt 還原 TrainState。

        Args:
            step:            要還原的 step；None → latest
            reference_state: 用於建構 abstract template 的 reference state；
                             必須在「網路初始化後、訓練前」用同樣的 init 流程
                             造出來，shape/dtype 與 saved state 一致即可。
                             若為 None，會嘗試用零模板做 lenient restore
                             （只能保證 dict 結構，不保證 NamedTuple 型別）。

        Why 需要 reference_state：
            orbax PyTreeRestore 無法從磁碟還原 Python class 結構，只能還原
            「leaves + tree structure 描述」。若想還原成原本的 TrainState
            NamedTuple（而不是純 dict），必須提供一個型別匹配的 abstract
            template。POC 訓練流程的 reference state 是 model.init() 結果，
            零成本可拿到。
        """
        target_step = self.latest_step() if step is None else int(step)
        if target_step is None:
            raise FileNotFoundError(
                f"directory {self._directory} 內找不到任何 ckpt，無法 restore"
            )

        if reference_state is None:
            # Lenient mode：不傳 abstract，orbax 會還原成巢狀 dict
            restored = self._mgr.restore(
                target_step,
                args=ocp.args.Composite(**{self._ITEM_NAME: ocp.args.PyTreeRestore()}),
            )
            raw = restored[self._ITEM_NAME]
            logger.warning(
                "ckpt.restore() 未提供 reference_state；回傳的是 dict 而非 TrainState。"
                " 建議在 production 流程一律提供 reference_state。"
            )
            return raw  # type: ignore[return-value]

        if not isinstance(reference_state, TrainState):
            raise TypeError(
                "reference_state 必須是 ckpt.TrainState，"
                f"收到 {type(reference_state).__name__}"
            )

        # reference_state 直接作為 orbax 的 template：它負責 (1) 還原成原本的
        # NamedTuple 型別而非 dict、(2) 指定預期 shape/dtype。
        # 此處曾有一層 `_make_abstract_template`，宣稱把 leaf 轉成 ShapeDtypeStruct
        # 以省下 restore 期間的記憶體。它從未生效——`is_leaf=lambda x: not
        # hasattr(x, "shape")` 對 root 求值時，TrainState 自己沒有 `.shape`，
        # 整棵樹因此被當成單一 leaf 原樣回傳（實測 `out is state`）。
        # 移除它是行為保持的。真要那個記憶體優化是另一件事，且會改變 restore 行為。
        restored = self._mgr.restore(
            target_step,
            args=ocp.args.Composite(
                **{self._ITEM_NAME: ocp.args.PyTreeRestore(reference_state)},
            ),
        )
        state = restored[self._ITEM_NAME]
        # orbax 偶爾會把 0-d int32 → numpy；統一轉回 jnp 確保下游 jax-friendly
        return state

    def latest_step(self) -> int | None:
        """已落盤步數中 step 值最大的那個；空目錄回 None。

        Note: orbax 原生 `latest_step()` 回的是「最後寫入」的 step，與
              save 順序耦合（若 caller 亂序 save 100→250→200，會回 200）。
              訓練 resume 場景下我們要的是「max step value」（單調訓練語意），
              因此這裡顯式取 `max(all_steps)`，行為更直覺。
        """
        steps = self.all_steps()
        return max(steps) if steps else None

    def all_steps(self) -> list[int]:
        """所有保留下來的 step（升冪）。"""
        return sorted(int(s) for s in self._mgr.all_steps())

    @property
    def directory(self) -> Path:
        return self._directory

    def close(self) -> None:
        """釋放 orbax 後端資源；測試 cleanup 用。"""
        # orbax CheckpointManager 沒有顯式 close API；wait_until_finished
        # 確保所有 pending async op 完成（同步 mode 是 no-op）
        self._mgr.wait_until_finished()


def verify_params_tree(init_params: Any, restored_params: Any) -> int:
    """驗證 restore 的參數樹與 model.init 重建的樹（path + shape）完全一致。

    Why:
      eval 端的 lenient restore（reference_state=None）不驗結構；若 eval --config
      與 ckpt 的訓練 config 不符（例：tau-off config 評 tau-on ckpt），Flax apply
      會忽略 ckpt 多出的參數、silent 跑殘缺 forward——不會報錯，只會給錯數字。

    Returns leaf 數（供 caller 印 log）。錯配時 raise ValueError 列出差異。
    """
    def _tree_index(tree: Any) -> dict[str, tuple]:
        return {
            jax.tree_util.keystr(kp): tuple(getattr(leaf, "shape", ()))
            for kp, leaf in jax.tree_util.tree_leaves_with_path(tree)
        }

    init_t = _tree_index(init_params)
    ckpt_t = _tree_index(restored_params)
    if init_t != ckpt_t:
        missing = sorted(set(init_t) - set(ckpt_t))
        extra = sorted(set(ckpt_t) - set(init_t))
        shape_bad = sorted(
            k for k in set(init_t) & set(ckpt_t) if init_t[k] != ckpt_t[k]
        )
        raise ValueError(
            "ckpt 參數樹與 config 重建的模型不一致（config/ckpt 錯配？）：\n"
            f"  config 有、ckpt 缺: {missing[:5]}{' …' if len(missing) > 5 else ''}\n"
            f"  ckpt 有、config 缺: {extra[:5]}{' …' if len(extra) > 5 else ''}\n"
            f"  shape 不符: {shape_bad[:5]}{' …' if len(shape_bad) > 5 else ''}"
        )
    return len(ckpt_t)


#: 設為 "1" 時，「無指紋」由警告升級為失敗。驗收 job 與 CI 不得騎弱路徑。
FINGERPRINT_STRICT_ENV = "PILNN_CKPT_FINGERPRINT_STRICT"


def _check_model_fingerprint(ckpt_dir: Path, model: Any) -> dict[str, Any]:
    """比對 eval 端建構與訓練時記錄的建構；回傳可直接寫進 eval 產物的 provenance。

    Why 需要這一層：`verify_params_tree` 只比 path + shape，**改 forward 卻不改
    參數樹**的旗標對它完全隱形（`disable_cross_attention` 兩側都是 14546 個參數、
    樹全等，而同一份 params 的輸出差 4.6e-2）。`configs/exp_245_b1_les_T50.toml`
    就設了那個旗標，所以這不是假想的失敗模式。

    舊 ckpt 沒有指紋，且**無法回溯補上**——artifacts_dir 只落 summary/eval_history/
    ledger，沒有 config 副本，訓練當時的建構值不存在於任何地方。那條路因此只能
    警告並在產物上蓋印記；`PILNN_CKPT_FINGERPRINT_STRICT=1` 讓它變成失敗。
    """
    summary_path = ckpt_dir.parent / "summary.json"
    current = model_fingerprint(model)

    def _unverified(reason: str) -> dict[str, Any]:
        if os.environ.get(FINGERPRINT_STRICT_ENV) == "1":
            raise ValueError(
                f"{reason}——{FINGERPRINT_STRICT_ENV}=1 下不接受未驗證的 ckpt。\n"
                f"  ckpt_dir={ckpt_dir}\n  summary={summary_path}")
        print(f"[ckpt][WARN] 建構指紋未驗證：{reason}")
        print("[ckpt][WARN]   → eval 產物將標記 fingerprint_verified=false；"
              "改 forward 但不改參數樹的旗標（如 disable_cross_attention）未受檢查")
        return {"fingerprint_verified": False, "reason": reason,
                "eval_construction": current}

    if not summary_path.exists():
        return _unverified(f"找不到 {summary_path}（訓練早於指紋機制，或 artifacts_dir 未對齊）")

    # 壞掉的 summary 要吵——靜默當成「舊 ckpt」會把解析失敗偽裝成正常的弱路徑。
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"summary.json 無法解析: {summary_path}: {exc}") from exc

    # 讀到**別人的** summary 比讀不到更糟——那會給出假綠。summary 自己記了
    # ckpt_dir，拿它確認這份 summary 確實描述正在 restore 的這個 ckpt。
    recorded_dir = summary.get("ckpt_dir")
    if recorded_dir is not None and Path(recorded_dir).resolve() != ckpt_dir.resolve():
        raise ValueError(
            "summary.json 描述的不是這個 checkpoint——artifacts_dir 未對齊：\n"
            f"  summary 記載 ckpt_dir={recorded_dir}\n  實際 restore={ckpt_dir}")

    recorded = summary.get("model_construction")
    if not recorded:
        return _unverified(f"{summary_path} 無 model_construction 欄位（訓練早於指紋機制）")

    problems = fingerprint_diff(recorded, current)
    if problems:
        raise ValueError(
            "eval 的模型建構與訓練時不符——restore 會安靜給出錯誤數字。\n"
            f"  ckpt_dir={ckpt_dir}\n  summary={summary_path}\n  差異：\n"
            + "\n".join(f"    - {p}" for p in problems))

    return {"fingerprint_verified": True, "reason": None}


#: 設為 "1" 時，「訓練端沒記 norm_stats」由警告升級為失敗。
NORM_STATS_STRICT_ENV = "PILNN_NORM_STATS_STRICT"

#: 同一份 sensor 檔配同一個 time_stride 會逐位元重現，容差只留給 float32/64 往返。
_NORM_STATS_RTOL = 1e-9


def verify_norm_stats(artifacts_dir: str | Path, norm_stats: dict,
                      *, dataset_index: int = 0) -> dict[str, Any]:
    """比對 eval 端的正規化常數與訓練時記錄的；回傳可寫進 eval 產物的 provenance。

    **第三道閘門，擋參數樹與建構指紋都看不見的東西。** `norm_stats` 既不是參數也不是
    建構值，而是**從 sensor 檔推導**出來的：`data.py` 載入時對 strided 後的子序列算
    per-channel mean/std。eval 端重載 sensor 時若 `time_stride` 與訓練不同，就會得到
    另一組常數，然後拿它反正規化模型輸出（`evaluate.py:94`）——整批數字系統性偏移，
    而現有兩道閘門全都放行。

    實測該偏移的尺度（K=100 主線 sensor 檔）：stride 8 讓 `v_std` 差 0.11%、
    stride 20 差 **2.17%**。`exp_snap_st20_*` 那 1.39% 的系統性底線就是這麼來的。

    舊 ckpt **無法回溯保護**：`summary.json` 在 2026-09-14 之前沒有這個欄位，而訓練
    當時的常數不存在於任何地方。那條路警告並在產物蓋 `norm_stats_verified: false`；
    `PILNN_NORM_STATS_STRICT=1` 讓它變成失敗。

    不符則**直接拋錯**——那不是弱路徑，是已知會給出錯誤數字的路徑。
    """
    artdir = Path(artifacts_dir)
    summary_path = artdir / "summary.json"
    current = {k: float(v) for k, v in norm_stats.items()}

    def _unverified(reason: str) -> dict[str, Any]:
        if os.environ.get(NORM_STATS_STRICT_ENV) == "1":
            raise ValueError(
                f"{reason}——{NORM_STATS_STRICT_ENV}=1 下不接受未驗證的正規化常數。\n"
                f"  artifacts_dir={artdir}\n  summary={summary_path}")
        print(f"[ckpt][WARN] 正規化常數未驗證：{reason}")
        print("[ckpt][WARN]   → eval 產物將標記 norm_stats_verified=false；"
              "eval 的 time_stride 若與訓練不同，反正規化常數就不同且無人檢查")
        return {"norm_stats_verified": False, "reason": reason,
                "eval_norm_stats": current}

    if not summary_path.exists():
        return _unverified(f"找不到 {summary_path}")
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"summary.json 無法解析: {summary_path}: {exc}") from exc

    recorded_all = summary.get("norm_stats")
    if not recorded_all:
        return _unverified(f"{summary_path} 無 norm_stats 欄位（訓練早於本機制）")
    if dataset_index >= len(recorded_all):
        raise ValueError(
            f"summary.json 只記了 {len(recorded_all)} 個 dataset 的 norm_stats，"
            f"但要取第 {dataset_index} 個——eval 與訓練的 dataset 數不一致")
    recorded = recorded_all[dataset_index]

    problems = []
    for k, want in recorded.items():
        if k not in current:
            problems.append(f"{k}: 訓練記 {want!r}，eval 端沒有這個通道")
            continue
        got = current[k]
        if abs(got - want) > _NORM_STATS_RTOL * max(abs(want), 1e-12):
            problems.append(f"{k}: 訓練 {want!r} vs eval {got!r} "
                            f"（相對差 {abs(got - want) / max(abs(want), 1e-12):.3e}）")
    if problems:
        raise ValueError(
            "eval 的正規化常數與訓練時不符——反正規化會安靜給出偏移的數字。\n"
            f"  artifacts_dir={artdir}\n  summary={summary_path}\n  差異：\n"
            + "\n".join(f"    - {p}" for p in problems)
            + "\n  最常見的成因是 eval 的 time_stride 與訓練不同："
            "norm_stats 是對 strided 後的子序列算的。"
            "\n  對齊 stride（evaluation_protocol 的 follow_training 模式），"
            "或明確說明為何要用另一組常數。")

    return {"norm_stats_verified": True, "reason": None}


#: reference params 只是形狀模板：查詢點數、init seed 與 re_norm 都不影響參數樹
#: （由 tests/test_restore_eval_params.py 釘住）。取固定值，免得呼叫端以為它們有意義。
_REFERENCE_QUERY_N = 8
_REFERENCE_INIT_SEED = 0
_REFERENCE_RE_NORM = 0.0


def reference_params_for(
    model: Any,
    sensor_vals: Any,
    sensor_pos: Any,
    sensor_time: Any,
) -> Any:
    """建 `restore_eval_params` 要的 reference params（參數樹模板）。

    Args:
        sensor_vals: `[T, K, C]`——與這次 eval 實際餵給 model 的是同一份（含
                     time stride 與任何切片）。**末維 C 是唯一會改變參數樹的維度。**
        sensor_pos:  `[K, 2]`
        sensor_time: `[T]`

    Why 存在：這段「PRNGKey → init_xy → init_t → model.init(7 args)」原本在四個
    eval / 診斷腳本各抄一次（`evaluate_exp245`、`evaluate_multi_re`、
    `diag_trainable_fourier`、`dump_cp_fields`），而且四份互不相同——有的 split
    rng 有的不 split，`re_norm` 有的讀 config 有的寫死 0.5，`T_total` 各自 fallback。
    實測那些差異**都不改變參數樹**：只有 `sensor_vals` 的通道數會。也就是說九行
    儀式裡承重的那一維，反而不在任何呼叫端的意圖裡。

    收進來之後呼叫端交出的是它真的知道的東西（model 與這次要餵的 sensor），而不是
    一組得猜對形狀的 dummy 輸入。猜錯 leading 維度時 `verify_params_tree` 看不見
    ——它只比 path 與 shape，兩者都不隨查詢點數改變。

    查詢點（`xy` / `t_q`）與 `re_norm` 在此取固定值：三個 operator 的 `__call__`
    都不用它們的**值**決定任何 param 形狀。`tests/test_restore_eval_params.py` 釘住這個等價性，
    哪天某個 arch 讓查詢影響參數樹，那個測試會先紅。
    """
    st = jnp.asarray(sensor_time)
    xy = jnp.zeros((_REFERENCE_QUERY_N, 2), dtype=jnp.float32)
    # 取 sensor_time[0] 而非 0.0：讓 dummy query 落在時間軸內，不必去想 decoder
    # 的 searchsorted 在域外會拿到什麼 index。值本身仍不影響參數樹。
    t_q = jnp.broadcast_to(st[:1], (_REFERENCE_QUERY_N,)).astype(jnp.float32)
    return model.init(
        jax.random.PRNGKey(_REFERENCE_INIT_SEED),
        jnp.asarray(sensor_vals),
        jnp.asarray(sensor_pos),
        _REFERENCE_RE_NORM,
        st,
        xy,
        t_q,
    )


def restore_eval_params(
    ckpt_dir: str | Path,
    ckpt: str | int,
    reference_params: Any,
    model: Any,
    *,
    verbose: bool = True,
) -> tuple[Any, int, dict[str, Any]]:
    """eval / 診斷腳本取 checkpoint params 的唯一入口。

    Args:
        ckpt_dir:         `<artifacts_dir>/checkpoints`
        ckpt:             `"latest"` 或明確 step
        reference_params: 由 `--config` 重建的 `model.init()` params，用來驗結構。
                          刻意設為必填——沒有它就無法擋 config/ckpt 錯配，而錯配
                          不會 crash，只會安靜給錯數字。
        model:            建出 `reference_params` 的那個 model 實例。同樣必填：
                          參數樹比對看不見「改 forward 卻不改樹」的旗標，要擋下
                          那一類錯配就必須拿得到建構值。可選的閘門＝裝飾性閘門。

    Returns `(params, restored_step, provenance)`。`provenance` 帶
    `fingerprint_verified`，呼叫端應原樣寫進 eval 產物——stdout 的警告會被滾走，
    而印記會跟著數字旅行到它進表格的那一天。

    Why 存在：這段「latest_step → None 檢查 → lenient restore → verify_params_tree」
    原本在三個 eval 腳本各抄一次，其中一份漏了後兩步。eval 產出直接變成論文數字，
    所以這裡一律 fail-fast：任何一步對不上就 raise，不 fallback、不取最近的 step。
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint 目錄不存在: {ckpt_dir}")

    mgr = CheckpointManager(directory=ckpt_dir, max_to_keep=3, save_interval_steps=1)
    available = mgr.all_steps()
    if not available:
        raise FileNotFoundError(f"無 ckpt 可 restore；dir={ckpt_dir}")

    if ckpt == "latest":
        target_step = max(available)
    else:
        target_step = int(ckpt)
        if target_step not in available:
            raise FileNotFoundError(
                f"指定的 step={target_step} 不在 {ckpt_dir}；可用: {sorted(available)}"
            )

    restored = mgr.restore(target_step, reference_state=None)
    n_leaves = verify_params_tree(reference_params, restored["params"])
    provenance = _check_model_fingerprint(ckpt_dir, model)
    restored_step = int(restored.get("step", target_step))
    if verbose:
        mark = "verified" if provenance["fingerprint_verified"] else "UNVERIFIED"
        print(f"[ckpt] restored step={restored_step} from {ckpt_dir} "
              f"(params tree: {n_leaves} leaves; 建構指紋: {mark})")
    return restored["params"], restored_step, provenance


__all__ = [
    "TrainState",
    "CheckpointManager",
    "verify_params_tree",
    "FINGERPRINT_STRICT_ENV",
    "reference_params_for",
    "restore_eval_params",
]

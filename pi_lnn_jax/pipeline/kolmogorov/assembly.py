"""建構期 —— 把 leaf module 組裝成 TrainingContext。

What: 資料載入 / physics closure / loss 組裝 / optimizer / jit step_fn /
      checkpoint manager，全部由 train_kolmogorov.py 逐字搬入。

硬性邊界（spec §4.2）：本模組只組裝依賴，不得含任何 runtime decision。
取樣時機、RNG split、RAR 觸發、weighting/AL 排程一律屬於 run.py。
若本檔開始回傳一堆 callback bundle，代表邊界已失敗，必須重切。

不搬 build_model：4792b93 已抽到 pi_lnn_jax/model_factory.py 並加哨兵測試，
本模組只呼叫它，不得長回本地副本（tests/test_model_factory.py 會紅）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.causal import causal_weights
from pi_lnn_jax.ckpt import CheckpointManager
from pi_lnn_jax.data import (
    load_dns,
    load_dns_from_path,
    load_multi_re_sensors,
    load_sensors_from_path,
)
from pi_lnn_jax.losses import al_constraint_value
from pi_lnn_jax.model_factory import build_model
from pi_lnn_jax.optimizers import accumulate_grads, build_optimizer, is_soap_available
from pi_lnn_jax.physics import make_ns_residual_fn, make_ns_residual_fn_baseline
from pi_lnn_jax.rollout import (
    check_extension_inputs,
    extend_sensor_sequence,
    pseudo_frame_times,
)
from pi_lnn_jax.pipeline.kolmogorov.config import (
    KolmogorovDataConfig,
    KolmogorovEffectiveConfig,
    KolmogorovPolicy,
)
from pi_lnn_jax.sensor_dropout import apply_sensor_dropout


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# re_norm 上界 Re_max 由 data_kwargs.re_norm_scale 提供（預設 1e4），見 _load_datasets


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Re runtime batch (Wave 5)
# ─────────────────────────────────────────────────────────────────────────────

class ReBatch(NamedTuple):
    """Per-Re runtime args bundled as JAX Pytree。

    Wave 5: 取代舊版 loss_fn / step_fn 透過 closure capture sensor + nu + norm_stats
    的反模式。Pytree 結構讓 jax.jit 把整個 ReBatch 視為單一 dynamic arg，
    multi-Re dataset 切換不會觸發 retrace。

    Shape contract（multi-Re training 跨 Re 必須一致）:
      sensor_vals: [T, K, C]
      sensor_pos:  [K, 2]
      sensor_time: [T]
      其他都是 scalar
    """
    sensor_vals: jnp.ndarray   # [T, K, C]
    sensor_pos: jnp.ndarray    # [K, 2]
    sensor_time: jnp.ndarray   # [T]
    re_norm: jnp.ndarray       # scalar
    nu: jnp.ndarray            # scalar = 1.0 / Re
    u_mean: jnp.ndarray        # scalar
    u_std: jnp.ndarray         # scalar
    v_mean: jnp.ndarray        # scalar
    v_std: jnp.ndarray         # scalar
    p_mean: jnp.ndarray        # scalar (baseline 無 p → 0.0)
    p_std: jnp.ndarray         # scalar (baseline 無 p → 1.0)


def _make_re_batch(dataset: dict, dtype: Any = jnp.float32) -> ReBatch:
    """Convert one dataset dict (from load_multi_re_sensors) into ReBatch."""
    ns = dataset["norm_stats"]
    return ReBatch(
        sensor_vals=jnp.asarray(dataset["sensor_vals"], dtype=dtype),
        sensor_pos=jnp.asarray(dataset["sensor_pos"], dtype=dtype),
        sensor_time=jnp.asarray(dataset["sensor_time"], dtype=dtype),
        re_norm=jnp.asarray(dataset["re_norm"], dtype=dtype),
        nu=jnp.asarray(1.0 / float(dataset["re_value"]), dtype=dtype),
        u_mean=jnp.asarray(ns["u_mean"], dtype=dtype),
        u_std=jnp.asarray(ns["u_std"], dtype=dtype),
        v_mean=jnp.asarray(ns["v_mean"], dtype=dtype),
        v_std=jnp.asarray(ns["v_std"], dtype=dtype),
        p_mean=jnp.asarray(ns.get("p_mean", 0.0), dtype=dtype),
        p_std=jnp.asarray(ns.get("p_std", 1.0), dtype=dtype),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 連續-Re 物理正則輔助：norm_stats 內插表 + sampled ReBatch builder
# ─────────────────────────────────────────────────────────────────────────────

_CRP_STAT_KEYS = ("u_mean", "u_std", "v_mean", "v_std", "p_mean", "p_std")


def re_norm_scale_of(data: KolmogorovDataConfig) -> float:
    """`data_kwargs.re_norm_scale` —— re_norm 的對數底；預設 1e4 = 單-Re 既有行為。

    純 config 衍生量。`build_context` 與 `run.replay_schedule` 的最小 ctx 共用
    這一份，避免兩邊各抄一次預設值後悄悄漂移。
    """
    return float(data.re_norm_scale)


def _build_crp_interp(datasets: list[dict]) -> dict:
    """建連續-Re 物理用的 norm_stats 內插表（host numpy）。

    回 dict：re_norm_sorted [N]、各 stat 的 [N] 陣列（依 re_norm 排序）、re_norm_min/max。
    """
    re_norms = np.array([float(d["re_norm"]) for d in datasets], dtype=np.float64)
    order = np.argsort(re_norms)
    table: dict = {"re_norm_sorted": re_norms[order]}
    for k in _CRP_STAT_KEYS:
        default = 0.0 if k.endswith("mean") else 1.0
        vals = np.array([float(d["norm_stats"].get(k, default)) for d in datasets])
        table[k] = vals[order]
    table["re_norm_min"] = float(re_norms[order][0])
    table["re_norm_max"] = float(re_norms[order][-1])
    return table


# NOTE: 以 sampled re_norm' 造 ReBatch 的 `_make_re_batch_crp` 已搬到
# `pi_lnn_jax/pipeline/kolmogorov/run.py` —— 它是 per-step 取樣的一部分（執行期），
# 留在建構期會讓入口腳本必須 import 一個 assembly 的私有名。
# 本檔只保留建構期的內插表 `_build_crp_interp`；兩者共用 `_CRP_STAT_KEYS`。


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def effective_sensor_query_points(requested: int, population: int) -> int:
    """實際可用的 sensor mini-batch 大小；population 不足時回 0（全取）。

    `num_sensor_query_points` 的語意是「每步**最多**取這麼多個 sensor 時空點」。
    抽樣走 replace=False，設定值大於 population（訓練實際餵入的 frames × K）
    就是硬錯誤，而 population 會隨每個縮短時間軸的 campaign 變動——cross-Re 的
    K=10、T=11 快照密度、90% 間歇都撞到同一件事。把判斷放在各 config 生成器
    裡等於每次都要記得重算一次，漏掉就是 45 秒後才爆的 job。

    population 不足時全取是唯一合理行為（抽樣本身失去意義），不是使用者設錯，
    故降級而非拋錯；降級會由呼叫端明確印出，不靜默改設定。
    """
    if population <= 0:
        raise ValueError(f"sensor query population 必須為正，收到 {population}")
    if requested and requested > population:
        return 0
    return requested




def assert_rar_pool_is_large_enough(rar_freq: int, rar_pool_size: int,
                                   n_collo_end: int,
                                   exploration_ratio: float = 0.2) -> None:
    """RAR 開著但 pool 裝不下 top-k —— 會在第一次觸發時才炸，擋在建構期。

    `rar_sample` 取 `n_top = round(n_select × (1 − exploration_ratio))` 個
    top-residual 點，而 `lax.top_k(pool, n_top)` 要求 `n_top <= pool_size`。
    **預設值本身就不相容**：`rar_pool_size=512` 配主線 `n_collo=1024` 需要 819 個，
    所以照預設打開 RAR 必定失敗——但要等到 `rar_warmup` 之後的第一個觸發步才炸，
    那時已經跑了幾百步。擋在這裡，設定錯當場知道。

    `exploration_ratio` 由 `curriculum.rar_exploration_ratio` 提供（預設 0.2）。
    """
    if rar_freq <= 0:
        return
    n_top = max(1, round(int(n_collo_end) * (1.0 - exploration_ratio)))
    if rar_pool_size >= n_top:
        return
    raise ValueError(
        f"rar_freq={rar_freq} 開啟 RAR，但 rar_pool_size={rar_pool_size} < "
        f"top-k 需要的 {n_top}（= round({n_collo_end} × {1.0 - exploration_ratio})）。"
        f"lax.top_k 會在第一次觸發時拋錯。rar_pool_size 建議取 n_collo 的 8–32 倍"
        f"（此處為 {8 * int(n_collo_end)}–{32 * int(n_collo_end)}）。"
    )


def assert_soap_betas_are_used(soap_b1, soap_b2, optimizer: str, base_optimizer: str) -> None:
    """宣告了 SOAP betas 卻沒走 SOAP —— 那兩個值會被完全忽略，大聲擋下。

    SOAP + Schedule-Free 只存在於 sbatch 的 CLI 旗標，TOML 沒有任何鍵能開它
    （`run.optimizer` 的相容預設是 "adam"）。於是「只用 config 檔跑訓練」會靜默拿到
    純 Adam，而 config 裡的 `soap_betas` 被完全忽略、不發任何警告。已跑的數字不受
    影響（都帶 CLI 旗標），受害的是照 `CLAUDE.md` §5 重現的人。

    擋在建構期而非 config 解析期：只讀 config 取 model kwargs 是正當用途，
    不該被迫假裝成一次訓練。

    預設值從 policy 取而不寫死——預設雙源正是本 repo 已立案的技術債之一。
    """
    default = (KolmogorovPolicy._defaults["run.soap_b1"],
               KolmogorovPolicy._defaults["run.soap_b2"])
    if (soap_b1, soap_b2) == default:
        return
    # `optimizer="adam"` 時 `build_optimizer` 根本不看 `base_optimizer`
    # （optimizers.py 的 adam 路徑），所以「base 是 soap」不足以放行。
    if optimizer in ("soap", "schedule_free"):
        return
    raise ValueError(
        f"config 宣告了 SOAP betas（b1={soap_b1}, b2={soap_b2}）但 optimizer 解析為 "
        f"{optimizer!r}/{base_optimizer!r} —— 兩個值都不會生效。SOAP 只能由 CLI "
        "開啟：--optimizer schedule_free --base_optimizer soap"
        "（注意：即使正確開了 SOAP，schedule_free 仍會強制內層 b1=0.0 以免動量套兩次，"
        "所以 b1 在任何路徑上都不生效；那是刻意的，見 chapter02 §Optimisation Stack）"
        "（見 scripts/slurm/train_exp.sbatch.tmpl）。"
    )


def _assert_declared_npz_matches(declared: list, loaded: dict, json_path) -> None:
    """`sensor_npzs` 是純裝飾的設定鍵——320 份 config 設了它，沒有任何載入程式讀它。

    實際的 NPZ 由 sensor JSON 的 `meta["dns_values_npz"]` 決定。所以把
    `sensor_npzs` 指到另一份檔（例如換成 noisy 版本）會**完全沒效果、沒警告**，
    而 provenance 還會忠實記錄那個從未被開啟的路徑。

    這裡不改變解析來源（JSON 仍是權威），只在兩者不一致時大聲失敗。目前全庫
    316 組宣告全部一致，故不影響任何既有 config。
    """
    if not declared:
        return
    from pathlib import Path as _Path

    want = _Path(str(declared[0])).name
    got = _Path(str(loaded.get("npz_path", ""))).name
    if want and got and want != got:
        raise ValueError(
            f"config 的 sensor_npzs[0] 宣告 {want!r}，但實際載入的是 {got!r}"
            f"（由 {json_path} 的 meta['dns_values_npz'] 決定）。"
            "sensor_npzs 不參與解析，改它不會換檔——要換 NPZ 請改 sensor JSON 的 meta。"
        )

def _load_datasets(data: KolmogorovDataConfig, multi_re: bool) -> list[dict]:
    """Return list of dataset dicts，length=1 or N。

    每個 dict 包含 sensor_vals/sensor_pos/sensor_time/norm_stats/re_value/re_norm。
    """
    json_paths = data.sensor_jsons
    re_values = data.re_values
    time_strides = data.time_strides  # Wave 5: per-Re stride
    # re_norm 上界 Re_max；預設 1e4 → 單-Re 既有行為不變
    re_norm_scale = float(data.re_norm_scale)

    if not json_paths:
        raise ValueError("TOML data_kwargs.sensor_jsons 為空，無法載入 sensor data")

    if multi_re:
        if len(json_paths) != len(re_values):
            raise ValueError(
                f"multi_re 模式: sensor_jsons ({len(json_paths)}) 與 re_values "
                f"({len(re_values)}) 長度不符"
            )
        # Wave 5: 若 TOML 給 time_strides list，per-Re 對齊；否則全部用預設 stride=2
        stride_arg = time_strides if time_strides else 2
        if isinstance(stride_arg, list) and len(stride_arg) != len(json_paths):
            raise ValueError(
                f"time_strides ({len(stride_arg)}) 與 sensor_jsons ({len(json_paths)}) 長度不符"
            )
        return load_multi_re_sensors(
            json_paths, re_values, time_stride=stride_arg, re_norm_scale=re_norm_scale
        )

    # 單 Re：只取第一份 sensor + Re；time_strides[0] 優先，否則預設 stride=2
    single_stride = int(time_strides[0]) if time_strides else 2
    d = load_sensors_from_path(json_paths[0], time_stride=single_stride)
    _assert_declared_npz_matches(data.sensor_npzs, d, json_paths[0])
    re_value = float(re_values[0])
    d["re_value"] = re_value
    d["re_norm"] = float(np.log(re_value) / np.log(re_norm_scale))
    return [d]


#: 截斷後「不存在」的 sensor 幀所用的時間哨兵。decoder 的 idx 是
#: `sum(sensor_time <= t_q) - 1`，把尾段推到這個值就等於那些幀不曾出現在
#: decode 端（encode 是 causal scan，截點的 h_states 本來就不受後面的幀影響）。
_CUT_SENTINEL_TIME = 1e30


def cut_sensor_time(sensor_time: jnp.ndarray, cut_idx: jnp.ndarray) -> jnp.ndarray:
    """把 `cut_idx` 之後的 sensor 時刻換成哨兵值，讓 decoder 看不到那些幀。

    Why 不直接切短陣列：shape 必須固定，否則每個 cut 值都會觸發 jit retrace。
    """
    frame = jnp.arange(sensor_time.shape[0])
    return jnp.where(frame <= cut_idx, sensor_time,
                     jnp.asarray(_CUT_SENTINEL_TIME, sensor_time.dtype))


def check_physics_t_max(phys_t_max: float, data_t_max: float, T_total: float) -> None:
    """physics 時間外推的前提檢查（`physics_t_max` = 0 時是 no-op）。

    Why: 這個鍵只用來把 collocation **延伸**到資料時窗之外（外推段只有 PDE
    residual、無 data loss）。兩個前提若不成立，訓練不會 crash、只會安靜地
    訓出另一個東西：
      - 小於資料時窗 → 實際是縮小 physics 覆蓋，那是 time marching 的職責；
      - 大於 `T_total` → `temporal_phase_anchor` 週期為 T_total，t 與 t−T_total
        會落在同一個相位，外推段與訓練段混疊。
    """
    if phys_t_max <= 0.0:
        return
    if phys_t_max < data_t_max:
        raise ValueError(
            f"physics_t_max={phys_t_max} 小於資料時窗上界 {data_t_max}；本鍵只用於把 "
            "collocation 延伸到資料之外，縮小時窗請用 time marching"
        )
    if phys_t_max > T_total:
        raise ValueError(
            f"physics_t_max={phys_t_max} 超過 model.T_total={T_total}；"
            "temporal_phase_anchor 週期為 T_total，外推段會與訓練段相位混疊。"
            "請把 T_total 一併設為外推後的總時窗"
        )
    print(f"  physics collocation 時窗: [0, {phys_t_max}]"
          f"（資料時窗 [0, {data_t_max}]；外推段只有 PDE residual）")


def _load_dns_for_eval(data: KolmogorovDataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """eval truth DNS。

    若 data_kwargs.dns_paths 提供則用 dns_paths[0]，否則 fallback 到 pi-lnn Re=1000
    的 hard-coded 路徑（單-Re POC 既有相容）。

    重要：訓練期 eval 只比對 d0（datasets[0]），故 dns_paths[0] 必須對應 re_values[0]，
    且須與 sensors 同一 DNS realization（否則 metrics 比對到錯誤流場）。
    """
    dns_paths = data.dns_paths
    if dns_paths:
        return load_dns_from_path(dns_paths[0], time_stride=1)
    return load_dns(time_stride=2)


# Model factory 的唯一實作在 pi_lnn_jax.model_factory；scripts/evaluate_exp245.py
# 匯入同一個物件，訓練端與 eval 端因此不可能漂移（哨兵測試見 tests/test_model_factory.py）。


# ─────────────────────────────────────────────────────────────────────────────
# Loss assembly: closure that returns jit-able loss_fn
# ─────────────────────────────────────────────────────────────────────────────

def _build_loss_fn(
    model,
    ns_fn,
    poisson_fn,
    use_poisson: bool,
    use_al: bool,
    al_rho: float,
    w_poisson: float,
    cont_gradnorm: bool = False,
    gauge_loss_weight: float = 0.0,
    al_constraint_mode: str = "mse",
    t_early_weight: float = 1.0,
    t_early_threshold: float = 0.05,
    T_total: float = 5.0,
    use_causal: bool = False,
    sensor_channel_weights: tuple = (),
    use_continuous_re_physics: bool = False,
    continuous_re_physics_weight: float = 1.0,
    autoreg_pseudo_times=None,     # np.ndarray | None；建構期算好的 pseudo 幀時刻
    autoreg_rounds: int = 1,
    ns_fn_baseline=None,   # B0/PINN baseline NS 殘差（liquid 路徑不用）
):
    """Return loss_fn(params, cx, cy, ct, task_weights, al_lambda, w_phys_now, re_batch)。

    Wave 5 signature: 把 sensor/normalization/Re 相關全部從 closure 移到 ReBatch
    runtime arg。multi-Re training 切換 dataset 不再 silent fail。

    task_weights = [w_data, w_ns_u, w_ns_v]（cont_gradnorm 時為四項，末項為 continuity）
    al_lambda    = scalar (從 ALState.lambda_ 取)
    w_phys_now   = scalar (physics weight schedule，每 step 從 Python 算)
    re_batch     = ReBatch (per-Re sensor + nu + norm_stats)
    """
    # 線性阻尼係數直接取自 model 實例，不另拉 config 管線：它是 model dataclass 欄位，
    # 已被 model_fingerprint 涵蓋（§7.2），從同一處讀才不可能與指紋脫節。
    # 舊模型無此欄位 → 0.0 → residual 內的 `if drag_alpha` 不成立 → 逐位元不變。
    _drag_alpha = float(getattr(model, "drag_alpha", 0.0))
    LiquidOperator_class = model.__class__
    # encode/decode methods: 主 LiquidOperator 有；B0/B2 baseline 無 → fallback
    has_encode = hasattr(LiquidOperator_class, "encode") and hasattr(
        LiquidOperator_class, "decode_query"
    )

    def loss_fn(
        params_, cx, cy, ct, task_weights, al_lambda, w_phys_now, re_batch, causal_eps,
        sensor_idx=None, re_batch_crp=None, sensor_input_mask=None, sensor_cut_idx=None,
    ):
        sensor_vals = re_batch.sensor_vals
        sensor_pos = re_batch.sensor_pos
        sensor_time = re_batch.sensor_time
        re_norm = re_batch.re_norm

        # train-time sensor dropout（denoising 語意）：mask 只作用「模型輸入」，
        # sensor_target（下方）仍用未 mask 的原始 sensor_vals（監督真值）。
        # mask 在 jit 外每 step 預生成、shape 固定 [K] → 不 retrace（比照 sensor_idx）。
        # 註：CRP physics 路徑（re_batch_crp）與 GradNorm probe 不套此 mask（次要路徑）。
        sensor_vals_in = (
            apply_sensor_dropout(sensor_vals, sensor_input_mask)
            if sensor_input_mask is not None else sensor_vals
        )

        # branch 自迴歸：把序列延伸到 physics_t_max，讓外推段的 branch 狀態會隨時間變。
        # 關閉（預設）時 vals_enc/time_enc 就是原物件，圖與舊版逐字相同。
        if autoreg_pseudo_times is not None and has_encode:
            vals_enc, time_enc = extend_sensor_sequence(
                model, params_, sensor_vals_in, sensor_pos, re_norm, sensor_time,
                autoreg_pseudo_times, autoreg_rounds,
                encode_method=LiquidOperator_class.encode,
                decode_method=LiquidOperator_class.decode_query,
            )
        else:
            vals_enc, time_enc = sensor_vals_in, sensor_time

        # decode 端可見的時間軸。兩個 opt-in 都作用在這裡：
        #   sensor_cut_idx —— 截點之後的幀推到哨兵值 → branch 卡在截點、dt>0 有真值可學；
        #   time_enc       —— 自迴歸延伸後的時間軸（含 pseudo 幀）。
        # 兩者不得同時開（build_context fail-fast），否則無法歸因。
        # 皆關閉時原樣傳遞，jit 圖與舊版逐字相同。encode 端見上方 vals_enc/time_enc。
        # 註：CRP physics 路徑與 GradNorm probe 不套（次要路徑，同 dropout 的處理）。
        sensor_time_dec = (
            time_enc if sensor_cut_idx is None
            else cut_sensor_time(time_enc, sensor_cut_idx)
        )


        # derive sensor query tensors from sensor_pos/time/vals (shapes are static)
        T, K, C = sensor_vals.shape
        xy_sensor_q = jnp.broadcast_to(sensor_pos[None, :, :], (T, K, 2)).reshape(T * K, 2)
        t_sensor_q = jnp.broadcast_to(sensor_time[:, None], (T, K)).reshape(T * K)
        sensor_target = sensor_vals.reshape(T * K, C)

        # sensor mini-batch：用預先取樣的 sensor_idx（shape 固定 → jit 不 retrace）
        # encode 仍全量（Wave 4 encode-once），只縮 decode query 的點數。
        if sensor_idx is not None:
            xy_sensor_q = xy_sensor_q[sensor_idx]
            t_sensor_q = t_sensor_q[sensor_idx]
            sensor_target = sensor_target[sensor_idx]

        # Wave 4 perf: encode once + decode-only for sensor pred + PDE
        if has_encode:
            h_states = model.apply(
                params_, vals_enc, sensor_pos, re_norm, time_enc,
                method=LiquidOperator_class.encode,
            )
            pred = model.apply(
                params_, xy_sensor_q, t_sensor_q, h_states, sensor_time_dec, sensor_pos,
                method=LiquidOperator_class.decode_query,
            )
        else:
            # B0/B2 baselines: 直接走 __call__ (內部沒 encode 拆分)
            pred = model.apply(
                params_, sensor_vals_in, sensor_pos, re_norm, sensor_time,
                xy_sensor_q, t_sensor_q,
            )
            h_states = None  # for arch without encode

        # t_early upweighting：對齊 pi-lnn t_early_weight=10.0 / t_early_threshold=0.05
        # 前期時間段（t < T_total × threshold）的 sensor loss 乘 t_early_weight
        # Why: 強化 temporal anchor，讓 latent state 在早期 t 收斂更好 → physics 更穩
        # decode_query 恆出 [N,3](u,v,p)；pred[:, :C] 切前 C 個對齊 sensor_target(C=2 baseline / 3 uvp)
        # per-channel 權重（順序 u,v,p）；空 → 等權（與既有 bit-identical）。
        if sensor_channel_weights and len(sensor_channel_weights) < C:
            raise ValueError(
                f"sensor_channel_weights 長度 {len(sensor_channel_weights)} < channel 數 {C}"
            )
        ch_w = (jnp.asarray(sensor_channel_weights[:C], dtype=sensor_target.dtype)
                if sensor_channel_weights else jnp.ones((C,), dtype=sensor_target.dtype))
        if t_early_weight != 1.0:
            t_threshold = float(t_early_threshold) * float(T_total)
            t_mask = (t_sensor_q < t_threshold).astype(jnp.float32)
            per_point_w = t_mask * (float(t_early_weight) - 1.0) + 1.0  # [T*K]
            per_point_err = jnp.mean(ch_w * (pred[:, :C] - sensor_target) ** 2, axis=-1)  # [T*K]
            sensor_loss = jnp.mean(per_point_w * per_point_err)
        else:
            sensor_loss = jnp.mean(ch_w * (pred[:, :C] - sensor_target) ** 2)
        # ForcingPrior 動態取 (A, k_f)
        A_force, k_f_force = model.apply(
            params_, method=LiquidOperator_class.get_forcing,
        )
        if has_encode:
            needs_cont_points = use_causal or (use_al and al_constraint_mode == "signed_mean")
            if needs_cont_points:
                mom_u, mom_v, cont, mom_u_pp, mom_v_pp, cont_pp = ns_fn(
                    params_, h_states, cx, cy, ct, A_force, k_f_force,
                    sensor_pos, sensor_time_dec,
                    re_batch.nu, re_batch.u_mean, re_batch.u_std,
                    re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
                    drag_alpha=_drag_alpha,
                    return_per_point=True,
                )
                if use_causal:
                    # Wang2022 因果加權：以逐點殘差能量決定時間權重，
                    # 加權後的等效 physics 純量取代原始 mean 殘差（w_c 已 stop_gradient）。
                    r_pp = mom_u_pp ** 2 + mom_v_pp ** 2 + cont_pp ** 2
                    w_c = causal_weights(ct, r_pp, causal_eps)
                    mom_u_eff = jnp.mean(w_c * mom_u_pp ** 2)
                    mom_v_eff = jnp.mean(w_c * mom_v_pp ** 2)
                    cont_eff = jnp.mean(w_c * cont_pp ** 2)
                else:
                    mom_u_eff, mom_v_eff, cont_eff = mom_u, mom_v, cont
            else:
                mom_u, mom_v, cont = ns_fn(
                    params_, h_states, cx, cy, ct, A_force, k_f_force,
                    sensor_pos, sensor_time_dec,
                    re_batch.nu, re_batch.u_mean, re_batch.u_std,
                    re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
                    drag_alpha=_drag_alpha,
                )
                mom_u_eff, mom_v_eff, cont_eff = mom_u, mom_v, cont
                cont_pp = None
        else:
            # B0(vanilla)/PINN baseline: 全 __call__ + 標準 autodiff（無 h_states）
            needs_cont_points = use_causal or (use_al and al_constraint_mode == "signed_mean")
            if needs_cont_points:
                mom_u, mom_v, cont, mom_u_pp, mom_v_pp, cont_pp = ns_fn_baseline(
                    params_, sensor_vals_in, sensor_pos, re_norm, sensor_time,
                    cx, cy, ct, A_force, k_f_force,
                    re_batch.nu, re_batch.u_mean, re_batch.u_std,
                    re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
                    drag_alpha=_drag_alpha,
                    return_per_point=True,
                )
                if use_causal:
                    r_pp = mom_u_pp ** 2 + mom_v_pp ** 2 + cont_pp ** 2
                    w_c = causal_weights(ct, r_pp, causal_eps)
                    mom_u_eff = jnp.mean(w_c * mom_u_pp ** 2)
                    mom_v_eff = jnp.mean(w_c * mom_v_pp ** 2)
                    cont_eff = jnp.mean(w_c * cont_pp ** 2)
                else:
                    mom_u_eff, mom_v_eff, cont_eff = mom_u, mom_v, cont
            else:
                mom_u, mom_v, cont = ns_fn_baseline(
                    params_, sensor_vals_in, sensor_pos, re_norm, sensor_time,
                    cx, cy, ct, A_force, k_f_force,
                    re_batch.nu, re_batch.u_mean, re_batch.u_std,
                    re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
                    drag_alpha=_drag_alpha,
                )
                mom_u_eff, mom_v_eff, cont_eff = mom_u, mom_v, cont
                cont_pp = None
        # 連續-Re 物理正則（spec 2026-06-13）：在 sampled re_norm' 重編碼 + NS 殘差，
        # 加權併入物理 task（data loss / AL / causal 不受影響，仍只用 data-Re）。
        if use_continuous_re_physics:
            h_states_crp = model.apply(
                params_, re_batch_crp.sensor_vals, re_batch_crp.sensor_pos,
                re_batch_crp.re_norm, re_batch_crp.sensor_time,
                method=LiquidOperator_class.encode,
            )
            mom_u_c, mom_v_c, cont_c = ns_fn(
                params_, h_states_crp, cx, cy, ct, A_force, k_f_force,
                re_batch_crp.sensor_pos, re_batch_crp.sensor_time,
                re_batch_crp.nu, re_batch_crp.u_mean, re_batch_crp.u_std,
                re_batch_crp.v_mean, re_batch_crp.v_std,
                re_batch_crp.p_mean, re_batch_crp.p_std,
            )
            w_crp = float(continuous_re_physics_weight)
            mom_u_eff = mom_u_eff + w_crp * mom_u_c
            mom_v_eff = mom_v_eff + w_crp * mom_v_c
            cont_eff = cont_eff + w_crp * cont_c
        if use_poisson:
            # 不傳 drag_alpha：`poisson_residual` 沒有這個參數，傳了會 TypeError。
            # 也不該有——壓力 Poisson 由不可壓動量方程取散度導出，線性阻尼項
            # 貢獻 −α·div(u)，在該推導的前提下為 0。此路徑因 poisson_weight
            # 全庫皆 0 而從未執行，錯誤直到 chapter05 那條未來工作被實作才會浮現。
            poisson_r = poisson_fn(
                params_, h_states, cx, cy, ct,
                sensor_pos, sensor_time_dec,
                re_batch.u_mean, re_batch.u_std,
                re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
            )
        else:
            poisson_r = jnp.array(0.0, dtype=sensor_loss.dtype)
        # AL term（若 disable，al_lambda=0、al_rho=0 → 整項 0；同樣寫法不分支）
        if use_al:
            # AL 刻意用未加權的 `cont`，不是 `cont_eff`：λ 追的是**物理**連續性
            # 殘差，而 causal 權重是訓練排程的產物（Wang 2022 只談 PDE 殘差損失，
            # 不涵蓋約束項）。同括號的 momentum 與 cont_gn_term 走 `_eff`，兩者
            # 因此在 causal 開啟時差一個數量級——那是刻意的，不是漏改。
            # （`use_causal` 全庫皆 False，此差異目前不影響任何數字。）
            al_c = al_constraint_value(cont, cont_pp, mode=al_constraint_mode)
            # ρ 懲罰必須用逐點散度平方 mean(div²)=cont（不相消）。
            # signed_mean 的 al_c=mean(div) 會正負相消，若拿來當 ρ penalty（(mean div)²）
            # 則逐點 continuity 完全失約束（div 暴增）——故 signed_mean 的 ρ 項改用 cont。
            # 共享 λ 的逐點 equality ALM：λ·mean(div) + ρ/2·mean(div²)。
            al_penalty = cont if al_constraint_mode == "signed_mean" else al_c ** 2
            al_cont_term = al_lambda * al_c + 0.5 * al_rho * al_penalty
        else:
            al_c = jnp.array(0.0, dtype=sensor_loss.dtype)
            al_cont_term = jnp.array(0.0, dtype=sensor_loss.dtype)
        if gauge_loss_weight > 0.0 and has_encode:
            # gauge 錦定：把預測 p 在 collocation batch 的空間均值錨定到 0
            # （對齊 DNS Poisson k=0 模態設 0 慣例）。多一次 decode-only 前向。
            uvp_c = model.apply(
                params_, jnp.stack([cx, cy], axis=-1), ct, h_states,
                sensor_time_dec, sensor_pos, method=LiquidOperator_class.decode_query,
            )
            p_c_phys = uvp_c[:, 2] * re_batch.p_std + re_batch.p_mean
            gauge_term = gauge_loss_weight * jnp.mean(p_c_phys) ** 2
        else:
            gauge_term = jnp.array(0.0, dtype=sensor_loss.dtype)
        # continuity 預設不進 weighted physics task：純由 AL term（al_cont_term）約束。
        # cont_gradnorm 時改走第四個 GradNorm task，AL 項恆 0（兩者互斥，見 config.py）。
        # 用 cont_eff 而非 cont：與同一括號內的 mom_*_eff 同一個量（causal 加權與 CRP
        # 都已套上）。混用會讓 continuity 與 momentum 在 causal 開啟時差一個數量級。
        cont_gn_term = (
            task_weights[3] * cont_eff
            if cont_gradnorm else jnp.array(0.0, dtype=sensor_loss.dtype)
        )
        total = (
            task_weights[0] * sensor_loss
            + w_phys_now * (
                task_weights[1] * mom_u_eff
                + task_weights[2] * mom_v_eff
                + cont_gn_term
            )
            + al_cont_term
            + w_poisson * poisson_r
            + gauge_term
        )
        return total, (sensor_loss, mom_u, mom_v, cont, poisson_r, al_c)

    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# Per-task gradient norms for GradNorm
# ─────────────────────────────────────────────────────────────────────────────

def _build_grad_norm_fn(
    model,
    ns_fn,
    ns_fn_baseline=None,   # B0/PINN baseline NS 殘差（liquid 路徑不用）
    ref_param_path: tuple = ("temporal_encoder",),
    sensor_channel_weights: tuple = (),
    grad_accum: int = 1,
    cont_gradnorm: bool = False,
    t_early_weight: float = 1.0,
    t_early_threshold: float = 0.05,
    T_total: float = 5.0,
    use_causal: bool = False,
):
    """Return jit-able compute_3task_grad_norms(params, cx, cy, ct, re_batch, causal_eps) -> [3] L2 norms。

    **probe 必須與主 loss 量同一個東西。** GradNorm 的前提是 w_i = G_0/G_i 平衡的是
    *實際被最佳化的* task；probe 若算另一個量，權重就是照著一個虛構的量調出來的。
    因此 `t_early_weight` / `use_causal` 與主 loss 同步傳入，`grad_accum>1` 的子取樣
    改為系統性均勻取樣（見 data_loss_only / compute_3task_grad_norms 內的說明）。

    刻意**不**對齊的一處：probe 走全量 `T*K` 網格而主 loss 走隨機 mini-batch。
    mini-batch 是均勻抽樣，其期望值就是全量網格均值，所以全量版是同一個量的
    零變異數估計，不是偏差。

    Wave 5 signature: sensor / norm_stats / nu / re_norm 全部從 closure 移到 re_batch
    runtime arg；multi-Re 訓練 grad-norm 用「當前 Re」算，不會 silent 用 d0。

    ⚠️ 下列「v6 修正」描述的是**本函式簽章的預設值，而非實際生效的路徑**。
    build 端（見本檔 gn_ref_path）一律覆蓋為 ("query_decoder", "trunk_out")／
    ("trunk_out",)，所有歷史 Kolmogorov production run 皆以此訓練；
    ("temporal_encoder",) 從未被任何 run 使用。改預設值屬行為變更，見該處 NOTE。

    Reference layer 選擇原則（v6 修正；僅描述上述未使用的預設值）：
      - 原始 ("query_decoder", "trunk_out") 會因 Fourier feature 的二階空間微分
        (2π·k)² amplification 讓 G_phys >> G_data（k=16 → 放大 ~10,000×），
        導致 GradNorm 反向運作（降低 physics weight 而非提升）。
      - 改用 ("temporal_encoder",) 全子樹：
        1. temporal_encoder 只處理 sensor 資料，不接觸 physics query 的 Fourier 編碼。
        2. G_data_enc ≈ K·T=10100 sensor query 梯度累積（大）；
           G_phys_enc ≈ 1024 collocation 梯度（小）。
        3. 使 G_data > G_phys at encoder → GradNorm 正確提升 physics weight。
      - 與 pi-lnn 的 trunk_out 基準在「效果層面」對齊（兩者都讓 w_phys 從 0.057 升至 0.12+）。

    若 ref_param_path 在 params 內找不到，fallback 到整個 params（warning 一次）。
    """
    # 阻尼須與 _build_loss_fn 同源：GradNorm 用不同 residual 會把任務權重算錯。
    _drag_alpha = float(getattr(model, "drag_alpha", 0.0))
    LiquidOperator_class = model.__class__
    has_encode = hasattr(LiquidOperator_class, "encode") and hasattr(
        LiquidOperator_class, "decode_query"
    )

    def data_loss_only(p, re_batch, n_sub=0):
        sensor_vals = re_batch.sensor_vals
        sensor_pos = re_batch.sensor_pos
        sensor_time = re_batch.sensor_time
        re_norm = re_batch.re_norm
        T, K, C = sensor_vals.shape
        xy_sensor_q = jnp.broadcast_to(sensor_pos[None, :, :], (T, K, 2)).reshape(T * K, 2)
        t_sensor_q = jnp.broadcast_to(sensor_time[:, None], (T, K)).reshape(T * K)
        sensor_target = sensor_vals.reshape(T * K, C)
        # n_sub>0：只 decode n_sub 個 (T*K) 點（大 K 下全量 T*K decode 會 OOM）。
        # **系統性均勻取樣，不是取前綴。** 攤平是 time-major，取前綴等於只取最早的
        # n_sub/K 個時刻——而那正是誤差最大的一段，會讓 G_data 系統性偏高。那是偏差
        # 不是變異數，EMA 平均不掉。linspace 取樣不消耗 RNG（不動 ledger）、在時間軸上
        # 均勻，且其 (index mod K) 會循環故不會鎖定 QR-pivot 的前幾個高重要度感測器。
        if n_sub and 0 < n_sub < T * K:
            sub = jnp.linspace(0, T * K - 1, n_sub).astype(jnp.int32)
            xy_sensor_q = xy_sensor_q[sub]
            t_sensor_q = t_sensor_q[sub]
            sensor_target = sensor_target[sub]
        if has_encode:
            h_states = model.apply(p, sensor_vals, sensor_pos, re_norm, sensor_time,
                                    method=LiquidOperator_class.encode)
            pred = model.apply(p, xy_sensor_q, t_sensor_q, h_states, sensor_time, sensor_pos,
                               method=LiquidOperator_class.decode_query)
        else:
            pred = model.apply(p, sensor_vals, sensor_pos, re_norm, sensor_time,
                               xy_sensor_q, t_sensor_q)
        if sensor_channel_weights and len(sensor_channel_weights) < C:
            raise ValueError(
                f"sensor_channel_weights 長度 {len(sensor_channel_weights)} < channel 數 {C}"
            )
        ch_w = (jnp.asarray(sensor_channel_weights[:C], dtype=sensor_target.dtype)
                if sensor_channel_weights else jnp.ones((C,), dtype=sensor_target.dtype))
        # t_early 加權與主 loss 同步（`_build_loss_fn` 同名區塊逐字對應）。缺了它，
        # G_data 量的是未加權的 data loss，而被最佳化的是加權版。
        if t_early_weight != 1.0:
            t_threshold = float(t_early_threshold) * float(T_total)
            t_mask = (t_sensor_q < t_threshold).astype(jnp.float32)
            per_point_w = t_mask * (float(t_early_weight) - 1.0) + 1.0
            per_point_err = jnp.mean(ch_w * (pred[:, :C] - sensor_target) ** 2, axis=-1)
            return jnp.mean(per_point_w * per_point_err)
        return jnp.mean(ch_w * (pred[:, :C] - sensor_target) ** 2)

    def physics_components(p, cx, cy, ct, re_batch, causal_eps):
        A_f, k_f_f = model.apply(p, method=LiquidOperator_class.get_forcing)
        if has_encode:
            h_states = model.apply(p, re_batch.sensor_vals, re_batch.sensor_pos,
                                    re_batch.re_norm, re_batch.sensor_time,
                                    method=LiquidOperator_class.encode)
            if use_causal:
                # 與主 loss 同步：加權後的等效殘差取代原始 mean。缺了它，G_phys 量的是
                # 未加權殘差，而被最佳化的是 causal 加權版——分母錯了，權重就錯。
                _, _, _, mom_u_pp, mom_v_pp, cont_pp = ns_fn(
                    p, h_states, cx, cy, ct, A_f, k_f_f,
                    re_batch.sensor_pos, re_batch.sensor_time,
                    re_batch.nu, re_batch.u_mean, re_batch.u_std,
                    re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
                    drag_alpha=_drag_alpha, return_per_point=True)
                w_c = causal_weights(ct, mom_u_pp ** 2 + mom_v_pp ** 2 + cont_pp ** 2,
                                     causal_eps)
                return (jnp.mean(w_c * mom_u_pp ** 2), jnp.mean(w_c * mom_v_pp ** 2),
                        jnp.mean(w_c * cont_pp ** 2))
            return ns_fn(p, h_states, cx, cy, ct, A_f, k_f_f,
                         re_batch.sensor_pos, re_batch.sensor_time,
                         re_batch.nu, re_batch.u_mean, re_batch.u_std,
                         re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
                         drag_alpha=_drag_alpha)
        if use_causal:
            _, _, _, mom_u_pp, mom_v_pp, cont_pp = ns_fn_baseline(
                p, re_batch.sensor_vals, re_batch.sensor_pos,
                re_batch.re_norm, re_batch.sensor_time,
                cx, cy, ct, A_f, k_f_f, re_batch.nu,
                re_batch.u_mean, re_batch.u_std,
                re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
                drag_alpha=_drag_alpha, return_per_point=True)
            w_c = causal_weights(ct, mom_u_pp ** 2 + mom_v_pp ** 2 + cont_pp ** 2,
                                 causal_eps)
            return (jnp.mean(w_c * mom_u_pp ** 2), jnp.mean(w_c * mom_v_pp ** 2),
                    jnp.mean(w_c * cont_pp ** 2))
        return ns_fn_baseline(p, re_batch.sensor_vals, re_batch.sensor_pos,
                              re_batch.re_norm, re_batch.sensor_time,
                              cx, cy, ct, A_f, k_f_f, re_batch.nu,
                              re_batch.u_mean, re_batch.u_std,
                              re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
                              drag_alpha=_drag_alpha)

    def ns_u_loss_only(p, cx, cy, ct, re_batch, causal_eps):
        return physics_components(p, cx, cy, ct, re_batch, causal_eps)[0]

    def ns_v_loss_only(p, cx, cy, ct, re_batch, causal_eps):
        return physics_components(p, cx, cy, ct, re_batch, causal_eps)[1]

    def cont_loss_only(p, cx, cy, ct, re_batch, causal_eps):
        return physics_components(p, cx, cy, ct, re_batch, causal_eps)[2]

    def _get_subtree(grads):
        """Resolve ref_param_path 在 grads 內；找不到 fallback 到 full grads。"""
        try:
            sub = grads["params"]
            for k in ref_param_path:
                sub = sub[k]
            return sub
        except (KeyError, TypeError):
            return grads

    @jax.jit
    def compute_3task_grad_norms(params_, cx, cy, ct, re_batch, causal_eps):
        # grad_accum>1：probe 只取 1/M 的點避免全量 decode OOM。
        # **系統性均勻取樣，不是取前綴。** collocation 在 RAR 關閉時是 iid 均勻，取前綴
        # 無偏；但 RAR 開啟時 rar_sample 回傳的是 [top_k(|residual|) 降冪 | random]，
        # 取前綴等於只拿殘差最大的那些點 → G_phys 系統性偏高。linspace 取樣在兩種
        # 情形下都正確：iid 下仍無偏，RAR 下按比例橫跨 top-k 與 exploration 兩段。
        if grad_accum > 1:
            n0 = cx.shape[0] // grad_accum
            sub = jnp.linspace(0, cx.shape[0] - 1, n0).astype(jnp.int32)
            cx, cy, ct = cx[sub], cy[sub], ct[sub]
            n_data_sub = n0          # data task 取同樣點數（取樣方式見 data_loss_only）
        else:
            n_data_sub = 0
        norms = []
        # data: 不需 cx/cy/ct，但要 re_batch
        g_data = jax.grad(data_loss_only)(params_, re_batch, n_data_sub)
        ref_data = _get_subtree(g_data)
        leaves = jax.tree_util.tree_leaves(ref_data)
        sq = sum(jnp.sum(leaf ** 2) for leaf in leaves) if leaves else jnp.array(0.0)
        norms.append(jnp.sqrt(sq + 1e-12))
        # physics tasks：grad over cx/cy/ct
        _physics_tasks = ((ns_u_loss_only, ns_v_loss_only, cont_loss_only)
                          if cont_gradnorm else (ns_u_loss_only, ns_v_loss_only))
        for task_fn in _physics_tasks:
            g = jax.grad(task_fn)(params_, cx, cy, ct, re_batch, causal_eps)
            ref = _get_subtree(g)
            leaves = jax.tree_util.tree_leaves(ref)
            sq = sum(jnp.sum(leaf ** 2) for leaf in leaves) if leaves else jnp.array(0.0)
            norms.append(jnp.sqrt(sq + 1e-12))
        return jnp.stack(norms)

    return compute_3task_grad_norms


# ─────────────────────────────────────────────────────────────────────────────
# Construction-time context
# ─────────────────────────────────────────────────────────────────────────────

class TrainingContext(NamedTuple):
    """建構期產物 —— 只有依賴，沒有可變狀態。

    不持有 params / opt_state：params 由 model.init 在執行期產生（消耗主 RNG），
    opt_state 由其衍生，兩者皆屬 run 期的 TrainingState（spec §4.2）。
    """
    config: KolmogorovEffectiveConfig
    model: Any
    model_name: str
    datasets: list[dict]
    re_batches: list          # list[ReBatch]
    re_t_min_host: list[float]
    re_t_max_host: list[float]
    crp_interp: dict
    crp_re_norm_scale: float
    dns_u: Any
    dns_v: Any
    dns_t: Any
    ns_fn: Any
    poisson_fn: Any
    ns_fn_baseline: Any
    loss_fn: Any
    grad_norm_fn: Any         # None if weighting off
    gn_ref_path: tuple
    tx: Any
    opt_info: dict
    step_fn: Any              # jit 好的
    ckpt_mgr: Any
    artifacts_dir: Any        # Path；已 resolve，落 summary/eval_history/ledger
    ckpt_dir: Any             # Path；= artifacts_dir / "checkpoints"，唯一推導點
    use_poisson: bool
    T_total: float
    n_sensor_query: int       # assembly-derived clamp; requested value stays in config


def build_context(config: KolmogorovEffectiveConfig) -> TrainingContext:
    """EffectiveConfig → TrainingContext；數值組裝順序維持不變。

    此處不得出現 RNG split、取樣、排程或任何 state 初始化：
    model.init / tx.init / gradnorm_init / al_init / rar_init
    全部留在執行期（spec §4.2、§5.1）。
    """
    run = config.run
    loss = config.loss
    curriculum = config.curriculum

    # ── 產物落點：artifacts_dir / ckpt_dir 的**唯一**推導點 ──
    # run 期（restore / run_loop / finalize）一律讀 ctx.artifacts_dir / ctx.ckpt_dir，
    # 不各自再算一次 `Path(eff["artifacts_dir"]).resolve() / "checkpoints"`。
    # Why: 重複推導正是 build_model 燒過一次、現在得靠哨兵測試守的形狀，而
    # 「artifacts_dir 未對齊」是本專案記錄有案的 eval 失敗模式。
    # 目錄建立：artifacts_dir 在此自建（summary.json / eval_history.json /
    # rng_ledger.json 落這層）；ckpt_dir 由 CheckpointManager 以 parents=True 自建。
    artifacts_dir = Path(run.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = artifacts_dir / "checkpoints"

    # ── 啟動 banner（原 train_kolmogorov.main() 開頭逐字搬入）──
    # 印在載資料之前，與搬移前的 stdout 順序一致。
    print("=" * 80)
    print(f"train_kolmogorov.py — config={Path(run.config_path).name}")
    print(f"  arch={run.arch}  optimizer={run.optimizer}")
    print(f"  steps={run.steps}  seed={run.seed}  artifacts={artifacts_dir}")
    print(f"  lr={run.learning_rate}  w_phys_final={loss.physics_weight}  "
          f"w_poisson={loss.poisson_weight}")
    print(f"  use_gradnorm={loss.use_gradnorm}  cont_gradnorm={loss.cont_gradnorm}  "
          f"use_al={loss.use_al}  "
          f"al_constraint_mode={loss.al_constraint_mode}  multi_re={run.multi_re}")
    print(f"  curriculum n_collo: {curriculum.n_collo_start} → {curriculum.n_collo_end} "
          f"(ramp={curriculum.n_collo_ramp})")
    print(f"  time_marching={curriculum.use_time_marching}  RAR freq={curriculum.rar_freq}")
    print("=" * 80)

    # ── 載入資料 ──
    datasets = _load_datasets(config.data, multi_re=run.multi_re)
    n_datasets = len(datasets)
    dns_u_full, dns_v_full, dns_t_full = _load_dns_for_eval(config.data)
    # DNS subsample 對齊 sensor_time（每 dataset 一致；用 [0] 的 stride）
    # stride 用 round((T_dns-1)/(T-1)) 而非 floor(T_dns/T)：floor 版在 T_dns 略大於
    # T 的整數倍時會 silent 截短 eval 視窗（例：DNS 201 幀、T=101 → floor stride=1
    # 只覆蓋前半段 t∈[0,2.5]）。此 eval 僅供訓練監控；正式數字走 evaluate_exp245。
    T_dataset = datasets[0]["sensor_vals"].shape[0]
    t_stride_dns = max(1, round((dns_u_full.shape[0] - 1) / max(1, T_dataset - 1)))
    dns_u = dns_u_full[::t_stride_dns][:T_dataset]
    dns_v = dns_v_full[::t_stride_dns][:T_dataset]
    dns_t = dns_t_full[::t_stride_dns][:T_dataset]
    _st0 = np.asarray(datasets[0]["sensor_time"], dtype=np.float64)
    _dt0 = np.asarray(dns_t, dtype=np.float64)
    if _dt0.shape[0] != _st0.shape[0] or not np.allclose(_dt0, _st0[: _dt0.shape[0]], atol=1e-6):
        print(f"WARNING: in-training eval DNS 時間軸與 sensor_time 未對齊 "
              f"(dns_t[{_dt0.shape[0]}] ∈ [{_dt0[0]:.3f},{_dt0[-1]:.3f}] vs "
              f"sensor_time[{_st0.shape[0]}] ∈ [{_st0[0]:.3f},{_st0[-1]:.3f}])；"
              f"監控指標僅供趨勢參考，正式數字以 evaluate_exp245 為準。")
    print(f"\nData: {n_datasets} dataset(s); DNS for eval [{dns_u.shape}], "
          f"t ∈ [{float(dns_t[0]):.2f}, {float(dns_t[-1]):.2f}]")
    for i, d in enumerate(datasets):
        print(f"  dataset[{i}] Re={d['re_value']:.0f}  sensor_vals={d['sensor_vals'].shape}  "
              f"u_std={d['norm_stats']['u_std']:.4f}")

    # Wave 5: 建立 per-Re ReBatch list（Python list of ReBatch，給 host 迴圈用）
    re_batches = [_make_re_batch(d) for d in datasets]
    d0 = datasets[0]
    T, K = d0["sensor_vals"].shape[0], d0["sensor_vals"].shape[1]
    _nsq_req = curriculum.n_sensor_query_requested
    _nsq_eff = effective_sensor_query_points(_nsq_req, T * K)
    if _nsq_eff != _nsq_req:
        print(f"  sensor_query: 設定 {_nsq_req} > population {T * K}（{T} frames × K={K}）"
              f" → 降為全取")
    print(f"  sensor_query: {'full ' + str(T * K) if not _nsq_eff else 'mini-batch ' + str(_nsq_eff) + '/' + str(T * K)}")
    T_total = config.model.T_total

    # 以下四項原本寫在 main() 訓練 loop 的前置段；純 host 端推導、不碰 RNG，
    # 提前到建構期不改變任何取樣時序。
    # 預先在 host 算每個 Re 的 t_min/t_max 範圍（Python float，給 collocation sample 用）
    re_t_min_host = [float(np.asarray(rb.sensor_time)[0]) for rb in re_batches]
    re_t_max_host = [float(np.asarray(rb.sensor_time)[-1]) for rb in re_batches]
    # physics 時間外推（physics_t_max>0）的前提檢查：它只能**延伸**時窗，不能縮小
    # （縮小是 time marching 的職責）；且相位錨的週期是 T_total，短於 physics 時窗
    # 就會讓 t 與 t−T_total 混疊成同一個相位 —— 兩者都在此 fail-fast，不靜默容忍。
    check_physics_t_max(
        float(config.curriculum.physics_t_max), max(re_t_max_host), float(T_total))
    if config.curriculum.sensor_cut_min_frac > 0.0 and run.arch != "liquid":
        # 截斷只作用在 decode 端；B0/PINN 走 __call__（encode+decode 一體），
        # 同一個 sensor_time 會連 encode 一起截掉，語意不同 —— 不靜默套用。
        raise ValueError(
            f"sensor_cut_min_frac 只支援 arch=liquid（收到 {run.arch!r}）："
            "baseline 架構的 encode/decode 共用同一條 sensor_time，截斷語意不同")
    # 連續-Re 物理：norm_stats 內插表 + re_norm 取樣範圍（旗標 off 時建表無害、不被用）
    crp_interp = _build_crp_interp(datasets)
    crp_re_norm_scale = re_norm_scale_of(config.data)

    # ── 建模（只造 Module；model.init 消耗主 RNG，屬執行期）──
    model, model_name = build_model(run.arch, config.model.to_kwargs(), K_sensors=K)

    # ── Physics residual closures (Wave 5: sensor/nu/norm_stats 改 runtime args) ──
    ns_fn, poisson_fn = make_ns_residual_fn(model)
    # B0(vanilla)/PINN baseline 無 encode/decode_query → 全 __call__ + 標準 autodiff
    ns_fn_baseline = make_ns_residual_fn_baseline(model)
    use_poisson = loss.poisson_weight > 0.0
    print(f"  Pressure-Poisson: {'ON' if use_poisson else 'OFF'} "
          f"(w_poisson={loss.poisson_weight})")

    # ── Loss assembly (Wave 5: re_batch 變 runtime arg) ──
    # branch 自迴歸的 pseudo 幀時刻（建構期常數；shape 固定 → 不 retrace）
    _ar_dt = float(config.curriculum.autoreg_pseudo_dt)
    if _ar_dt > 0.0:
        if config.curriculum.sensor_cut_min_frac > 0.0:
            raise ValueError(
                "autoreg_pseudo_dt 與 sensor_cut_min_frac 不得同時開啟："
                "兩者都改寫 decode 端的時間軸，混用無法歸因")
        _ar_t_end = float(config.curriculum.physics_t_max) or max(re_t_max_host)
        autoreg_pseudo_times = check_extension_inputs(
            pseudo_frame_times(max(re_t_max_host), _ar_t_end, _ar_dt),
            max(re_t_max_host), int(config.curriculum.autoreg_rounds))
        print(f"  branch 自迴歸: {len(autoreg_pseudo_times)} 個 pseudo 幀 "
              f"({max(re_t_max_host)}→{_ar_t_end}, dt={_ar_dt})，"
              f"{config.curriculum.autoreg_rounds} 輪")
    else:
        autoreg_pseudo_times = None

    loss_fn = _build_loss_fn(
        model, ns_fn, poisson_fn,
        ns_fn_baseline=ns_fn_baseline,
        use_poisson=use_poisson,
        use_al=loss.use_al, cont_gradnorm=loss.cont_gradnorm, al_rho=loss.al_rho,
        w_poisson=loss.poisson_weight,
        gauge_loss_weight=loss.gauge_weight,
        sensor_channel_weights=tuple(loss.sensor_channel_weights),
        al_constraint_mode=loss.al_constraint_mode,
        t_early_weight=loss.t_early_weight,
        t_early_threshold=loss.t_early_threshold,
        T_total=config.model.T_total,
        use_causal=loss.use_causal,
        use_continuous_re_physics=loss.use_continuous_re_physics,
        continuous_re_physics_weight=loss.continuous_re_physics_weight,
        autoreg_pseudo_times=autoreg_pseudo_times,
        autoreg_rounds=int(config.curriculum.autoreg_rounds),
    )

    # ── Optimizer ──
    if run.optimizer == "soap" and not is_soap_available():
        print("[WARN] SOAP requested but soap_jax not installed → adam fallback "
              "(see optimizers.py log)", flush=True)
    # SOAP betas: CLI override > TOML > default 0.95/0.95
    soap_b1 = run.soap_b1
    soap_b2 = run.soap_b2
    # SOAP + Schedule-Free 只存在於 sbatch 的 CLI 旗標，TOML 沒有任何鍵能開它
    # （run.optimizer 的相容預設是 "adam"）。於是「只用 config 檔跑訓練」會靜默拿到
    # 純 Adam，而 config 裡的 soap_betas 被完全忽略、不發任何警告。已跑的數字不受
    # 影響（都帶 CLI 旗標），受害的是照 CLAUDE.md §5 重現的人。擋在建構期而非 config
    # 解析期：只讀 config 取 model kwargs 是正當用途，不該被迫假裝成一次訓練。
    assert_soap_betas_are_used(soap_b1, soap_b2, run.optimizer, run.base_optimizer)
    assert_rar_pool_is_large_enough(
        curriculum.rar_freq, curriculum.rar_pool_size, curriculum.n_collo_end,
        curriculum.rar_exploration_ratio)
    tx, opt_info = build_optimizer(
        name=run.optimizer,
        learning_rate=run.learning_rate,
        base_optimizer=run.base_optimizer,
        soap_precondition_frequency=run.soap_precondition_frequency,
        soap_b1=soap_b1,
        soap_b2=soap_b2,
        max_grad_norm=run.max_grad_norm,
        warmup_steps=config.schedule.warmup_steps,
        decay_steps=config.schedule.decay_steps,
        decay_rate=config.schedule.decay_gamma,
        min_learning_rate=config.schedule.min_learning_rate,
    )
    print(f"  SOAP betas: b1={soap_b1}, b2={soap_b2}")
    print(f"  Optimizer: {opt_info['name']}  lr_schedule={opt_info['lr_schedule']}  "
          f"fallback={opt_info['fallback_to']}")

    # ── Inter-task weighting 的「函式」部分（GradNorm | off）──
    # LRA 已於 2026-08-03 移除（`TrainState.lra_state` 恆 None，僅為 ckpt 格式相容；見 run.py）。
    # 只建 ref path 與 grad-norm 函式；gradnorm_init / al_init 產生的是
    # state，屬執行期，仍留在 main()。
    # NOTE: 此覆蓋值（trunk_out）與 _build_grad_norm_fn 預設（temporal_encoder，v6 修正）
    # 不同——所有歷史 Kolmogorov production run 皆以 trunk_out 訓練。變更此值屬行為變更，
    # 需 A/B 驗證（w_phys 是否貼 gradnorm_min_weight floor），見 knowledge/codebase/technical-debt.md。
    gn_ref_path = (
        ("query_decoder", "trunk_out") if run.arch == "liquid" else ("trunk_out",)
    )
    if loss.use_gradnorm:
        grad_norm_fn = _build_grad_norm_fn(
            model, ns_fn, ns_fn_baseline=ns_fn_baseline,
            ref_param_path=gn_ref_path,
            sensor_channel_weights=tuple(loss.sensor_channel_weights),
            grad_accum=int(curriculum.grad_accum_chunks),
            cont_gradnorm=loss.cont_gradnorm,
            t_early_weight=loss.t_early_weight,
            t_early_threshold=loss.t_early_threshold,
            T_total=config.model.T_total,
            use_causal=loss.use_causal,
        )
    else:
        grad_norm_fn = None

    # ── Compile main step (Wave 5: re_batch 變 runtime arg) ──
    # grad_accum_chunks>1：把 collocation(cx,cy,ct)+sensor_idx 切 M 塊、lax.scan 逐塊
    # value_and_grad 累積（峰值記憶體 ∝ 1/M），解大 K decode cross-attn OOM。
    # 數值上：data/physics 為 mean-loss → 精確等價；AL 的 C²(al_c) 為 per-chunk 近似。
    grad_accum = int(curriculum.grad_accum_chunks)

    @jax.jit
    def step_fn(params_, opt_state_, cx, cy, ct, task_w, al_lambda, w_phys_now, re_batch, causal_eps,
                sensor_idx=None, re_batch_crp=None, sensor_input_mask=None, sensor_cut_idx=None):
        if grad_accum > 1:
            if sensor_idx is None:
                raise ValueError(
                    "grad_accum_chunks>1 需要 n_sensor_query>0：sensor_idx=None 無法 reshape "
                    "分塊。請設 num_sensor_query_points>0 或 grad_accum_chunks=1。"
                )
            M = grad_accum
            chunks = (cx.reshape(M, -1), cy.reshape(M, -1), ct.reshape(M, -1),
                      sensor_idx.reshape(M, -1))

            def _vg(p, cx_c, cy_c, ct_c, sidx_c):
                return jax.value_and_grad(loss_fn, has_aux=True)(
                    p, cx_c, cy_c, ct_c, task_w, al_lambda, w_phys_now, re_batch, causal_eps,
                    sidx_c, re_batch_crp, sensor_input_mask, sensor_cut_idx,
                )
            grads, total, (sl, mu, mv, c, pr, al_c) = accumulate_grads(_vg, params_, chunks)
        else:
            (total, (sl, mu, mv, c, pr, al_c)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params_, cx, cy, ct, task_w, al_lambda, w_phys_now, re_batch, causal_eps,
                sensor_idx, re_batch_crp, sensor_input_mask, sensor_cut_idx,
            )
        updates, opt_state_ = tx.update(grads, opt_state_, params_)
        params_ = jax.tree_util.tree_map(lambda p, u: p + u, params_, updates)
        return params_, opt_state_, total, sl, mu, mv, c, pr, al_c

    # ── Checkpoint manager（restore / sanity check 屬執行期，見 run.restore）──
    ckpt_mgr = CheckpointManager(
        directory=ckpt_dir,
        max_to_keep=3,
        save_interval_steps=1,  # save_every gating 自行管控；mgr 不再做 interval
    )

    return TrainingContext(
        config=config,
        model=model,
        model_name=model_name,
        datasets=datasets,
        re_batches=re_batches,
        re_t_min_host=re_t_min_host,
        re_t_max_host=re_t_max_host,
        crp_interp=crp_interp,
        crp_re_norm_scale=crp_re_norm_scale,
        dns_u=dns_u,
        dns_v=dns_v,
        dns_t=dns_t,
        ns_fn=ns_fn,
        poisson_fn=poisson_fn,
        ns_fn_baseline=ns_fn_baseline,
        loss_fn=loss_fn,
        grad_norm_fn=grad_norm_fn,
        gn_ref_path=gn_ref_path,
        tx=tx,
        opt_info=opt_info,
        step_fn=step_fn,
        ckpt_mgr=ckpt_mgr,
        artifacts_dir=artifacts_dir,
        ckpt_dir=ckpt_dir,
        use_poisson=use_poisson,
        T_total=T_total,
        n_sensor_query=_nsq_eff,
    )

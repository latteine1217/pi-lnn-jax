"""建構期 —— 把 leaf module 組裝成 cylinder 的 TrainingContext。

What: 資料載入（含 `[K,T,C]→[T,K,C]` 轉置）/ 幾何 / 衍生純量 / 模型 /
      physics closure / per-task loss closures / GradNorm ref path /
      總 loss / optimizer / jit `step_fn`，全部由 `train_cylinder.py` 逐字搬入。

硬性邊界（spec §4、§5.1）：本模組只組裝依賴，不得含任何 runtime decision。
`model.init` 消耗主 RNG，`tx.init` / `gradnorm_init` / `al_init` 產生的是
可變狀態，四者全部留在執行期；`TrainingContext` 因此不持有 params / opt_state。
邊界由 tests/test_cylinder_assembly.py 以 AST 守（比照 Kolmogorov 的
tests/test_pipeline_assembly.py）。

GradNorm ref 子樹要注意兩段的歸屬不同：**path 常數**（`("temporal_encoder",)`）與
解析函式 `_get_subtree` 是建構期；印出「到底解析到哪個子樹」的 **probe** 要
`jax.grad(_data_loss)(params, …)`，吃真實 params，屬執行期（同 Kolmogorov：
`assembly` 出 `gn_ref_path`，`run.initialize` 才做 probe）。

spec §6 明確不修的既有行為，逐一標註於下方：
  1. `np.load(相對 CWD 路徑)` 繞過 `_resolve_data_path`
  2. 直建 `LiquidOperator(**CFG, …, torch_style_init=True)` 繞過 `model_factory`
  3. GradNorm ref path 硬寫 `("temporal_encoder",)`
三者都是 bit-identical 契約的一部分，改動屬行為變更，不得「順手改善」。

`build_context` 開頭只解構 typed effective config；下方數值建構順序維持不變。
"""
from __future__ import annotations

import sys
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.boundary import (
    CylinderGeometry, fluid_mask, wall_bc_loss,
    OscillatingGeometry, fluid_mask_moving,
    wall_bc_loss_moving,
)
from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.optimizers import build_optimizer, is_soap_available
from pi_lnn_jax.physics import make_ns_residual_fn
from pi_lnn_jax.pipeline.cylinder.config import CylinderEffectiveConfig


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# model CFG 對齊 PyTorch CEXP-002（configs/exp_cylinder_002_k100_bc.toml）：
# non-periodic + Fourier 嵌入 128 + d_model/operator_rank=256。
# 注意：train_cylinder_v1.py 的 baseline preset 是「非 parity 小模型探針」，不可沿用。
# T_total 不寫死於 CFG（依資料時間跨度於 build 時帶入；CEXP-002 用 20.0）。
CFG = dict(
    sensor_value_dim=2, d_model=256, d_time=16,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=2, token_attention_heads=4,
    num_query_mlp_layers=1, query_mlp_hidden_dim=256, operator_rank=256,
    decoder_attention_heads=4, use_temporal_anchor=True,
    temporal_anchor_harmonics=2, domain_length=1.0, fourier_embed_dim=128,
    # ── 以下三鍵刻意顯式，即使值與 models.py 的 dataclass 預設相同 ──
    # Why：CFG 未指定的鍵會落到 dataclass 預設，而那是**會變的**。2026-09-03 那次
    # 預設變更（rwf/tau False→True、scale 2.0→0.5）就這樣讓 cylinder 的參數量
    # 從 3,139,146 跳到 3,272,010（+4.2%），而唯一的提示是 test_cylinder_assembly
    # 的參數量 pin 報紅——建構值本身沒有任何地方會說話。顯式寫下來，cylinder 的
    # 模型規格就只由本檔決定。
    # ⚠️ 這三個值在 cylinder 上**從未被實驗驗證**（證據全來自 Kolmogorov），
    # 尤其 cfc_input_dependent_tau 連在 Kolmogorov 都測不到效果卻吃掉 98.8% 的
    # 參數增量。它們是裁決結果，不是結論——見 knowledge/codebase/technical-debt.md TD-4。
    use_rwf=True, cfc_input_dependent_tau=True, cfc_tau_mod_scale=0.5,
)

LO = LiquidOperator


# ─────────────────────────────────────────────────────────────────────────────
# Construction-time context
# ─────────────────────────────────────────────────────────────────────────────

class TrainingContext(NamedTuple):
    """建構期產物 —— 只有依賴，沒有可變狀態。

    不持有 params / opt_state：params 由 `model.init` 在執行期產生（消耗主 RNG），
    opt_state 由其衍生，兩者皆屬執行期（spec §5.1）。

    欄位只收「執行期或 eval 真的會讀」的東西。純建構中間量（`nu`、`t_thr_phys`、
    `u_inf_n`、`xy_q_full`、`tgt_full`…）已被 closure 捕獲，不重複掛在這裡——
    多一份可讀到的真相就多一個會漂移的來源。
    """
    config: CylinderEffectiveConfig
    npz: Any                  # np.load 結果；eval 的 DNS 欄位與尾列 provenance 由此讀
    sv_TKC: Any               # [T,K,C]（CfC 走時間軸）
    sp: Any                   # [K,2]
    st: Any                   # [T] 物理時間
    obs_mean: Any             # [C]
    obs_std: Any              # [C]
    re_norm: float            # RE_NORM = log(Re)/log(1e4)
    Lx: float
    Ly: float
    bcen: Any                 # body_center / base_center（eval 的靜態 body 遮罩用）
    br: float
    geom: Any                 # CylinderGeometry | OscillatingGeometry
    u_inf: float
    st0: float
    T_total: float
    tm_end: float
    K: int
    T: int
    n_sensor_q: int           # 已被 min(·, T*K) 夾過；迴圈只准讀這個
    t_q_full_np: Any          # host [T*K]；time-marching 分支的 valid 篩選用
    model: Any
    loss_fn: Any
    data_loss_fn: Any         # GradNorm ref 子樹 probe 用（probe 本身屬執行期）
    get_subtree: Any          # 同上；解析 gn_ref_path，找不到回原 grads
    gn_ref_path: tuple
    grad_norm_fn: Any         # jit 好的 compute_task_grad_norms
    tx: Any
    opt_info: dict
    step_fn: Any              # jit 好的


def build_context(config: CylinderEffectiveConfig) -> TrainingContext:
    """EffectiveConfig → TrainingContext；數值組裝順序維持不變。

    此處不得出現 RNG split、取樣或任何 state 初始化：
    `model.init` / `tx.init` / `gradnorm_init` / `al_init` 全部留在執行期。
    """
    run = config.run
    loss = config.loss
    curriculum = config.curriculum
    data = config.data
    bk = run.backend
    case, is_controlled = run.case, run.is_controlled
    pw, amp_scale = run.physics_weight_scale, run.controlled_amp_scale
    seed, steps, n_collo = run.seed, run.steps, run.n_collo
    n_sensor_q = run.n_sensor_query_requested
    bc_weight, bc_body_w, bc_n = loss.bc_weight, loss.bc_body_weight, loss.bc_n
    gradnorm_freq, gradnorm_min = loss.gradnorm_freq, loss.gradnorm_min
    gradnorm_max, al_rho = loss.gradnorm_max, loss.al_rho
    lr, opt_name, base_opt = run.learning_rate, run.optimizer, run.base_optimizer
    use_phys_denorm = run.use_physics_denormalization
    t_early_w, t_early_thr = loss.t_early_weight, loss.t_early_threshold
    use_tm, tm_start = curriculum.use_time_marching, curriculum.time_marching_start
    tm_warmup = config.time_marching_warmup_steps
    npz_key, npz_path, u_inf_cfg = data.npz_key, data.npz_path, data.u_inf

    # ── 啟動 banner（原 train_cylinder.main() 開頭逐字搬入）──
    # 印在載資料之前，與搬移前的 stdout 順序一致。
    print("=" * 80)
    print(f"train_cylinder.py (CEXP-002)  config={run.config_path}  backend={bk}")
    print(f"  steps={steps}  n_collo={n_collo}  n_sensor_q={n_sensor_q}  seed={seed}")
    print(f"  optimizer={opt_name}  base={base_opt}  lr={lr}")
    print(f"  bc_weight={bc_weight}  bc_body_weight={bc_body_w}  bc_n={bc_n}")
    print(f"  gradnorm_freq={gradnorm_freq}  use_physics_denorm={use_phys_denorm}")
    print(f"  gradnorm_min_weight={gradnorm_min}  gradnorm_max_weight="
          f"{gradnorm_max}{' (NO CAP)' if gradnorm_max <= 0.0 else ''}")
    print(f"  case={case}  is_controlled={is_controlled}  physics_weight_scale={pw}")
    print(f"  t_early_weight={t_early_w} (thr={t_early_thr})  "
          f"time_marching={use_tm} (start={tm_start}, warmup={tm_warmup} steps)")
    print("=" * 80)

    # ── 資料載入（欄位對齊 train_cylinder_v1.py）──
    # spec §6【不修 1】`np.load(相對 CWD 路徑)`：cylinder 刻意**繞過**
    # `_resolve_data_path`（PILNJAX_ROOT→PILNJAX_DATA_ROOT 的多處尋找）。改走專案解析會把
    # 「找不到就炸」變成「往別處找」，屬行為變更 → 不改。
    if not npz_path:
        print(f"[FATAL] data_kwargs.{npz_key} 未設定", file=sys.stderr)
        sys.exit(6)
    d = np.load(npz_path)
    sv = jnp.asarray(d["sensor_vals"], jnp.float32)        # [K,T,C] normalized
    sp = jnp.asarray(d["sensor_pos"], jnp.float32)         # [K,2]
    st = jnp.asarray(d["sensor_time"], jnp.float32)        # [T] 物理時間
    K, T, C = sv.shape
    n_sensor_q = min(n_sensor_q, T * K)                    # 上限為全量
    obs_mean = jnp.asarray(d["obs_mean"], jnp.float32)     # [C]
    obs_std = jnp.asarray(d["obs_std"], jnp.float32)
    Lx, Ly = float(d["Lx"]), float(d["Ly"])
    RE = float(d["re_value"])
    bcen = d["base_center"] if is_controlled else d["body_center"]
    br = float(d["body_radius"])

    # CfC 走時間軸：sensor_vals 需 [T,K,C]
    sv_TKC = jnp.transpose(sv, (1, 0, 2))
    RE_NORM = float(np.log(RE) / np.log(10000.0))
    st0 = float(st[0])
    T_total = float(st[-1] - st[0])
    # threshold 是 span 比例，比較對象 t_q_full 是絕對物理時間 → 門檻須加 st0 偏移
    # （st0=0 時等價舊行為；st0>0 的資料若不加，條件恆 False、t_early 錨定 silent 失效）
    t_thr_phys = st0 + t_early_thr * T_total    # normalized threshold → 物理時間門檻
    tm_end = float(st[-1])                       # time-marching t_max 上限 = 最後一個 sensor 時刻

    # u_inf：config cylinder_u_inf > 0 優先；否則 fallback npz bc_inflow_u；再無則 0.33
    if u_inf_cfg > 0.0:
        u_inf = u_inf_cfg
    else:
        u_inf = float(d["bc_inflow_u"]) if "bc_inflow_u" in d.files else 0.33

    # 物理黏性（C1 修正）：residual 在物理空間組裝（physics.py 導數 /Lx、速度 denorm 回 m/s、
    # 物理時間），故需物理 nu=u_inf·D/Re，非 1/Re。RealPDEBench cylinder 直徑 D=30mm 固定，
    # 改入流速度調 Re（Re=u_inf·D/ν；arxiv 2601.01829）；驗證 u_inf·D/Re≈9.85e-7≈水黏性。
    # 舊 nu=1/Re 高估黏性項 ~100-570×（依 u_inf）。physics 為弱 regularizer，預期對 KE 影響小。
    D_CYL = 0.03  # RealPDEBench cylinder diameter [m]（Re 參考長度）
    nu = u_inf * D_CYL / RE

    if is_controlled:
        # 動邊界：geometry 與 motion 皆由 dump 從場實測寫進 npz（RPB 不發布 body 軌跡；
        # osc_freq 是實測頻率，非檔名的 control_freq）。半徑用物理單位：body 是物理空間
        # 的正圓，Lx≠Ly 時單一 normalized 半徑會把它扭曲成橢圓。
        geom = OscillatingGeometry(
            base_center=(float(bcen[0]), float(bcen[1])),
            body_radius_phys=float(d["body_radius_phys"]), Lx=Lx, Ly=Ly, u_inf=u_inf,
            amp=float(d["osc_amp"]) * amp_scale, freq=float(d["osc_freq"]),
            phase=float(d["osc_phase"]),
            axis=tuple(float(z) for z in np.asarray(d["osc_axis"]).ravel()),
        )
    else:
        geom = CylinderGeometry(
            body_center=(float(bcen[0]), float(bcen[1])),
            body_radius=br, Lx=Lx, Ly=Ly, u_inf=u_inf,
        )

    # normalized BC 目標（比對 normalized 預測 vs normalized 目標，對齊 v1 80-82）
    u_inf_n = (u_inf - float(obs_mean[0])) / float(obs_std[0])
    u_zero_n = (0.0 - float(obs_mean[0])) / float(obs_std[0])
    v_zero_n = (0.0 - float(obs_mean[1])) / float(obs_std[1])

    print(f"  K={K} T={T} C={C}  Lx={Lx:.4f} Ly={Ly:.4f} Re={RE:.0f}  "
          f"body=({bcen[0]:.3f},{bcen[1]:.3f}) R={br:.3f}")
    print(f"  u_inf={u_inf:.4f} (norm {u_inf_n:.3f})  nu={nu:.3e}  "
          f"T_total={T_total:.4f} st0={st0:.4f}")
    if is_controlled:
        # 注意：上面那行的 R 是 npz 的 legacy RMS normalized 值，controlled 實際幾何用
        # body_radius_phys（物理正圓，見 OscillatingGeometry docstring）。
        print(f"  [controlled] R_phys={geom.body_radius_phys:.5f} "
              f"(D={2*geom.body_radius_phys:.4f})  amp={geom.amp:.5f} "
              f"(measured {float(d['osc_amp']):.5f} x scale {amp_scale}) "
              f"freq={geom.freq:.4f} phase={geom.phase:.4f} axis={geom.axis}")
        print(f"  [controlled] arm = {'STATIC (amp=0)' if geom.amp == 0.0 else 'MOVING'}"
              f"   traj_r2={float(d['traj_r2']):.3f}")

    # ── 建模（對齊 CEXP-002：non-periodic + radial relpos + no-SDF；不走 geo 家族）──
    # T_total 帶入資料實際時間跨度（temporal anchor 正規化；CFG 不寫死）。
    # spec §6【不修 2】直建 `LiquidOperator(**CFG, …)` 繞過 `model_factory.build_model`，
    # 且固定 `torch_style_init=True`。收斂到 factory 是行為變更（factory 的預設不同）→ 不改。
    # （model.init 消耗主 RNG，屬執行期，不在此。）
    model = LiquidOperator(
        **CFG, relpos_bias_mode="radial", periodic_domain=False, use_sdf_features=False,
        T_total=float(st[-1]),
        torch_style_init=True,   # 對齊 PyTorch nn.Linear init（Flax lecun_normal 大 √3/層→~9× over-energy）
    )

    # ── physics residual closure（A=0, k_f=0, Lx/Ly 物理尺度化）──
    ns_residuals, _poisson = make_ns_residual_fn(model)
    # ns_residuals 傳給的 u_mean/u_std...：物理 denorm 用 npz obs 統計（§7），p 無監督 → p_mean=0,p_std=1
    u_mean_v = float(obs_mean[0])
    u_std_v = float(obs_std[0])
    v_mean_v = float(obs_mean[1])
    v_std_v = float(obs_std[1])
    p_mean_v = 0.0
    p_std_v = 1.0

    # sensor query 全量（重建目標）：(time × pos)
    xy_q_full = jnp.broadcast_to(sp[None], (T, K, 2)).reshape(T * K, 2)
    t_q_full = jnp.broadcast_to(st[:, None], (T, K)).reshape(T * K)
    tgt_full = jnp.transpose(sv, (1, 0, 2)).reshape(T * K, 2)   # normalized target [.,2]

    # ── per-task loss closures（GradNorm ref 子樹 grad 用，各自重算 h=encode）──
    def _encode(p_):
        return model.apply(p_, sv_TKC, sp, RE_NORM, st, method=LO.encode)

    def _early_w(sq_idx):
        # t_early：前期時間步 (t < t_thr_phys) 的 data loss 乘 t_early_w（=1.0 時恆等）
        return jnp.where(t_q_full[sq_idx] < t_thr_phys, t_early_w, 1.0)[:, None]

    def _data_loss(p_, sq_idx):
        h = _encode(p_)
        pred = model.apply(p_, xy_q_full[sq_idx], t_q_full[sq_idx], h, st, sp, method=LO.decode_query)
        return jnp.mean(_early_w(sq_idx) * (pred[:, :2] - tgt_full[sq_idx]) ** 2)

    def _ns_components(p_, cx, cy, ct):
        """Return (ns_u, ns_v, cont) body-masked scalar losses。"""
        h = _encode(p_)
        mu, mv, c, mom_u, mom_v, cont = ns_residuals(
            p_, h, cx, cy, ct, 0.0, 0.0, sp, st,
            nu, u_mean_v, u_std_v, v_mean_v, v_std_v, p_mean_v, p_std_v,
            Lx=Lx, Ly=Ly, return_per_point=True,
        )
        m = fluid_mask_moving(cx, cy, ct, geom) if is_controlled else fluid_mask(cx, cy, geom)
        denom = jnp.maximum(jnp.sum(m), 1.0)
        ns_u = jnp.sum(m * mom_u ** 2) / denom
        ns_v = jnp.sum(m * mom_v ** 2) / denom
        cont_l = jnp.sum(m * cont ** 2) / denom
        return ns_u, ns_v, cont_l

    def _ns_u_loss(p_, cx, cy, ct):
        return _ns_components(p_, cx, cy, ct)[0]

    def _ns_v_loss(p_, cx, cy, ct):
        return _ns_components(p_, cx, cy, ct)[1]

    # ── GradNorm ref 子樹 = temporal_encoder（找不到 fallback full grads）──
    # spec §6【不修 3】硬寫 `("temporal_encoder",)`：與 Kolmogorov 的 `trunk_out` 不同，
    # 兩者都是各自的既有行為 → 不統一。
    REF_PATH = ("temporal_encoder",)

    def _get_subtree(grads):
        """Resolve REF_PATH 於 grads；找不到 fallback full grads（同 train_kolmogorov pattern）。"""
        try:
            sub = grads["params"]
            for k in REF_PATH:
                sub = sub[k]
            return sub
        except (KeyError, TypeError):
            return grads

    def _grad_norm(g):
        ref = _get_subtree(g)
        leaves = jax.tree_util.tree_leaves(ref)
        sq = sum(jnp.sum(leaf ** 2) for leaf in leaves) if leaves else jnp.array(0.0)
        return jnp.sqrt(sq + 1e-12)

    @jax.jit
    def compute_task_grad_norms(p_, cx, cy, ct, sq_idx):
        g_data = jax.grad(_data_loss)(p_, sq_idx)
        g_u = jax.grad(_ns_u_loss)(p_, cx, cy, ct)
        g_v = jax.grad(_ns_v_loss)(p_, cx, cy, ct)
        return jnp.stack([_grad_norm(g_data), _grad_norm(g_u), _grad_norm(g_v)])

    # ── total loss（task_weights 為 traced arg → jit 不每次重編譯）──
    def loss_fn(p_, cx, cy, ct, task_w, al_lambda, sq_idx, t_bc_hi, bc_in, bc_body, bc_slip, bc_bvel):
        h = _encode(p_)
        # data loss（sensor minibatch 重建；避免全量 T*K decode OOM；t_early 前期加權）
        pred = model.apply(p_, xy_q_full[sq_idx], t_q_full[sq_idx], h, st, sp, method=LO.decode_query)
        data_loss = jnp.mean(_early_w(sq_idx) * (pred[:, :2] - tgt_full[sq_idx]) ** 2)
        # physics（body-masked）
        mu, mv, c, mom_u, mom_v, cont = ns_residuals(
            p_, h, cx, cy, ct, 0.0, 0.0, sp, st,
            nu, u_mean_v, u_std_v, v_mean_v, v_std_v, p_mean_v, p_std_v,
            Lx=Lx, Ly=Ly, return_per_point=True,
        )
        m = fluid_mask_moving(cx, cy, ct, geom) if is_controlled else fluid_mask(cx, cy, geom)
        denom = jnp.maximum(jnp.sum(m), 1.0)
        ns_u = jnp.sum(m * mom_u ** 2) / denom
        ns_v = jnp.sum(m * mom_v ** 2) / denom
        cont_l = jnp.sum(m * cont ** 2) / denom
        # wall BC（固定權重）。cylinder：BC 點時間為 [0,1]，線性映射到物理時間 [st0, t_bc_hi]。
        # controlled：sample_wall_bc_moving 已用物理時間（body 位置與之相依）→ decode 直接用該時間，
        # body target = 移動壁速（normalized，見 §5 / boundary docstring）。
        if is_controlled:
            def decode_fn(xyt):
                return model.apply(p_, xyt[:, :2], xyt[:, 2], h, st, sp, method=LO.decode_query)
            bc_bvel_n = bc_bvel / u_inf
            bc_loss = wall_bc_loss_moving(decode_fn, bc_in, bc_body, bc_slip,
                                          bc_bvel_n, u_inf_n, v_zero_n, bc_body_w)
        else:
            def decode_fn(xyt):
                return model.apply(p_, xyt[:, :2], st0 + xyt[:, 2] * (t_bc_hi - st0), h, st, sp,
                                   method=LO.decode_query)
            bc_loss = wall_bc_loss(decode_fn, bc_in, bc_body, bc_slip,
                                   u_inf_n, u_zero_n, v_zero_n, bc_body_w)
        # continuity 純由 AL 約束（mse mode → al_c = cont_l），不進 weighted task
        al_cont_term = al_lambda * cont_l + 0.5 * al_rho * cont_l ** 2
        # physics 打折（controlled marginal r=0.148 → pw=0.3；cylinder pw=1.0 保持原行為）。
        # data / bc 不受 pw 影響，重建主力仍為 sensor data。
        total = (task_w[0] * data_loss
                 + pw * (task_w[1] * ns_u + task_w[2] * ns_v + al_cont_term)
                 + bc_weight * bc_loss)
        return total, (data_loss, ns_u, ns_v, cont_l, bc_loss)

    # ── optimizer（CEXP-002：schedule_free + soap；CPU 自動 fallback）──
    if base_opt == "soap" and not is_soap_available():
        print("[WARN] SOAP requested but soap_jax 未安裝 → adam fallback (見 optimizers.py)",
              flush=True)
    # 排程與 SOAP 超參從 config 讀。預設值與此處原本吃到的 build_optimizer 預設相同，
    # 所以既有 config 的行為不變；設了才偏離（A8：chapter02 描述的協定從未套用到本案）。
    tx, opt_info = build_optimizer(
        name=opt_name,
        learning_rate=lr,
        base_optimizer=base_opt,
        max_grad_norm=1.0,
        soap_b1=float(run.soap_b1),
        soap_b2=float(run.soap_b2),
        soap_precondition_frequency=int(run.soap_precondition_frequency),
        warmup_steps=int(run.lr_warmup_steps),
        decay_steps=int(run.lr_decay_steps),
        decay_rate=float(run.lr_decay_gamma),
        min_learning_rate=float(run.min_learning_rate),
    )

    @jax.jit
    def step_fn(p_, o_, cx, cy, ct, task_w, al_lambda, sq_idx, t_bc_hi, bc_in, bc_body, bc_slip, bc_bvel):
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p_, cx, cy, ct, task_w, al_lambda, sq_idx, t_bc_hi, bc_in, bc_body, bc_slip, bc_bvel)
        updates, o_ = tx.update(grads, o_, p_)
        p_ = jax.tree_util.tree_map(lambda a, b: a + b, p_, updates)
        return p_, o_, total, aux

    # 原本寫在訓練 loop 前置段（緊接 np_rng 之後）的純 host 轉換；不碰 RNG，
    # 提前到建構期不改變任何取樣時序。迴圈的 time-marching 分支用它篩 valid。
    t_q_full_np = np.asarray(t_q_full)

    return TrainingContext(
        config=config,
        npz=d,
        sv_TKC=sv_TKC,
        sp=sp,
        st=st,
        obs_mean=obs_mean,
        obs_std=obs_std,
        re_norm=RE_NORM,
        Lx=Lx,
        Ly=Ly,
        bcen=bcen,
        br=br,
        geom=geom,
        u_inf=u_inf,
        st0=st0,
        T_total=T_total,
        tm_end=tm_end,
        K=K,
        T=T,
        n_sensor_q=n_sensor_q,
        t_q_full_np=t_q_full_np,
        model=model,
        loss_fn=loss_fn,
        data_loss_fn=_data_loss,
        get_subtree=_get_subtree,
        gn_ref_path=REF_PATH,
        grad_norm_fn=compute_task_grad_norms,
        tx=tx,
        opt_info=opt_info,
        step_fn=step_fn,
    )

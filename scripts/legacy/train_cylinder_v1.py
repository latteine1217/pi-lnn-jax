"""Cylinder v1 可行性：geometry-aware cross-attention 對 flow-over-cylinder。

A/B：
  --mode baseline : radial + periodic + no-SDF（Kolmogorov-style attention）
  --mode geo      : vector + non-periodic + SDF（#1/#2/#3 geometry-aware）

loss = data(sensor 重建) + w_phys · NS殘差(forcing=0, Lx/Ly 物理尺度化, body 遮罩)
eval = 重建 DNS grid → field-L2 vs RealPDEBench DNS（held-out val 時刻）

限制（v1）：無 BC losses（inflow/no-slip/slip）、無 geometry graph → 非 pi-lnn-parity，
測「geometry-aware attention 是否對幾何案例有感」，非完整流場正確性。
"""
from __future__ import annotations
import argparse
import time
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.autodiff import make_fused_field_derivatives
from pi_lnn_jax.losses import al_init, al_update

CFG = dict(
    sensor_value_dim=2, d_model=96, d_time=16,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=2, token_attention_heads=4,
    num_query_mlp_layers=2, query_mlp_hidden_dim=128, operator_rank=96,
    decoder_attention_heads=4, use_temporal_anchor=True, T_total=1.0,  # T_total runtime override（見 build_model）
    temporal_anchor_harmonics=2, domain_length=1.0, fourier_embed_dim=128,  # RFF 預設（非週期，修 x=0≡x=L fold）
    # ── legacy 鎖定（2026-09-04）：這三鍵刻意釘在產生論文數字當時的值 ──
    # 本檔是論文 §7b / thesis tab:cyl_main 的 Re=1781, K=400, n=5 結果
    # （KE 相對誤差 5.92±0.92%）唯一的重現路徑。它原本一鍵不指定，於是模型
    # 規格由 models.py 的 dataclass 預設決定——2026-09-03 那批預設改動
    # （use_rwf / cfc_input_dependent_tau 皆 False→True、scale 2.0→0.5）
    # 因此會讓重跑建出不同的模型，論文數字變得不可重現。
    # 這裡不跟隨新預設，與 configs/exp_245_b3_les_T50.toml 的 legacy 鎖定同理：
    # 已發表數字綁在當時的行為上，重現路徑就該釘住當時的建構值。
    # 要用新預設做 cylinder，走現行主線 pi_lnn_jax/pipeline/cylinder/。
    use_rwf=False, cfc_input_dependent_tau=False, cfc_tau_mod_scale=2.0,
)


def build_model(mode, body_center, body_radius, attention_kind="scalar",
                fourier_embed_dim=None, t_total=None):
    cfg = dict(CFG)
    if fourier_embed_dim is not None:   # override：>0 走 RFF（非週期，修 cylinder x=0≡x=L fold）
        cfg["fourier_embed_dim"] = fourier_embed_dim
    if t_total is not None:   # 串接 runtime 時間窗：CFG T_total=1.0 是 Kolmogorov 殘留，
        cfg["T_total"] = t_total   # cylinder 實際 ~19.9，否則 temporal anchor sin/cos 繞 ~20 圈混疊
    geo = dict(
        relpos_bias_mode="vector", periodic_domain=False,
        use_sdf_features=True, body_center_x=float(body_center[0]),
        body_center_y=float(body_center[1]), body_radius=float(body_radius),
    ) if mode == "geo" else dict(
        relpos_bias_mode="radial", periodic_domain=True, use_sdf_features=False,
    )
    return LiquidOperator(**cfg, **geo, attention_kind=attention_kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "geo"], required=True)
    ap.add_argument("--fourier_embed_dim", type=int, default=None,
                    help="override CFG：>0 走 RFF（非週期，修週期 fold + 加高頻容量）；預設 None=用 CFG 的 periodic")
    ap.add_argument("--attention_kind", choices=["scalar", "vector"], default="scalar",
                    help="scalar=dot-product+bias(現行) | vector=PTv2 grouped vector attention")
    ap.add_argument("--data", default="data/cylinder_v1.npz")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_phys", type=float, default=0.01)
    ap.add_argument("--n_collo", type=int, default=1024)
    ap.add_argument("--n_sensor_q", type=int, default=2000)
    ap.add_argument("--use_bc", action="store_true", help="v2: 加 inflow/body-noslip/slip BC losses")
    ap.add_argument("--w_bc", type=float, default=0.1)
    ap.add_argument("--bc_body_w", type=float, default=2.0, help="body no-slip 額外加權（Zhu 2025）")
    ap.add_argument("--bc_n", type=int, default=128, help="每類 BC 取樣點數")
    ap.add_argument("--hard_wall", action="store_true",
                    help="上下 slip wall 改 hard BC ansatz（v=φ(y)·v_net 強制 v=0），取代 slip soft loss")
    # ── ALM continuity（取代 phys 內等權 cont²；λ dual ascent 強制 ∇·u→0）──
    ap.add_argument("--use_al", action="store_true",
                    help="continuity 改走 Augmented Lagrangian（λ·C+ρ/2·C²），從 phys 分離")
    ap.add_argument("--al_rho", type=float, default=1.0, help="ALM quadratic penalty ρ")
    ap.add_argument("--al_freeze_start", type=int, default=5000,
                    help="λ 開始 dual update 的步數（之前純 ρ/2·C² 暖機，λ=0）")
    ap.add_argument("--al_freeze_end", type=int, default=15000,
                    help="λ 停止 update 的步數（之後固定 λ 收斂）")
    ap.add_argument("--al_update_freq", type=int, default=100, help="λ dual update 頻率")
    ap.add_argument("--al_lambda_clip", type=float, default=10.0, help="λ 上限")
    ap.add_argument("--al_ema_momentum", type=float, default=0.9, help="C 的 EMA 平滑（collocation 隨機性）")
    # ── body 幾何處理升級（互斥）：hard gate ansatz vs Brinkman penalization ──
    ap.add_argument("--body_hard", action="store_true",
                    help="body no-slip 改 hard gate（u,v=clip(φ/scale,0,1)·net，pi-lnn Sukumar 2022），取代 body soft")
    ap.add_argument("--body_hard_scale", type=float, default=0.1, help="gate 過渡寬度 φ/scale")
    ap.add_argument("--brinkman", action="store_true",
                    help="沉浸邊界 penalization：移除 body mask，momentum 加 χ·u/η 強制 body 內 u→0，取代 body soft")
    ap.add_argument("--brinkman_eta", type=float, default=1e-2, help="Brinkman η（1/η = body 阻力強度）")
    ap.add_argument("--data_chunks", type=int, default=1,
                    help="data loss decode 分幾塊（大 K vector attention 省記憶體；保總採樣量，公平）")
    ap.add_argument("--freestream_ref", action="store_true",
                    help="Reynolds decomposition：u 參考用 freestream u_inf（非 sensor mean），全場背景→0")
    ap.add_argument("--dump_fields", default=None,
                    help="eval 後存第一個 val 時刻的 pred/dns 場（u,v,ω）到 npz，供畫圖")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    if args.body_hard and args.brinkman:
        sys.exit("[FATAL] --body_hard 與 --brinkman 互斥（兩種 body 處理擇一）")

    bk = jax.default_backend()
    if bk != "gpu" and not args.allow_cpu:
        print(f"[FATAL] backend={bk}; --allow-cpu to force"); sys.exit(5)
    print(f"=== Cylinder v1 mode={args.mode} backend={bk} ===")

    d = np.load(args.data)
    sv = jnp.asarray(d["sensor_vals"], jnp.float32)          # [K,T,C] normalized
    sp = jnp.asarray(d["sensor_pos"], jnp.float32)           # [K,2]
    st = jnp.asarray(d["sensor_time"], jnp.float32)          # [T]
    K, T, C = sv.shape
    obs_mean = jnp.asarray(d["obs_mean"], jnp.float32)       # [C]
    obs_std = jnp.asarray(d["obs_std"], jnp.float32)
    Lx, Ly = float(d["Lx"]), float(d["Ly"])
    RE = float(d["re_value"])
    bc = d["body_center"]; br = float(d["body_radius"])
    # BC 正規化目標（對齊 pi-lnn：比對 normalized 預測 vs normalized BC 目標）
    u_inf = float(d["bc_inflow_u"]) if "bc_inflow_u" in d.files else 0.33
    # 物理黏性（C1 修正，與 train_cylinder.py 一致）：residual 在物理空間（folx 導數 /Lx、
    # 速度 denorm、物理時間），故 nu=u_inf·D/Re（D=30mm RealPDEBench；Re=u_inf·D/ν），非 1/Re。
    D_CYL = 0.03  # RealPDEBench cylinder diameter [m]
    nu = u_inf * D_CYL / RE
    if args.freestream_ref:
        # Reynolds decomposition：u 參考由 sensor mean 改 freestream u_inf。
        # sensor_vals 已用 old obs_mean[0] normalize → 平移到 u_inf 參考（std 不變，減常數）。
        # 效果：全場背景(freestream)→0、inflow 擾動→0；physics 仍用 denorm 物理 u（精確）。
        old_um = float(obs_mean[0])
        shift = (u_inf - old_um) / float(obs_std[0])
        sv = sv.at[:, :, 0].add(-shift)
        obs_mean = obs_mean.at[0].set(u_inf)
        print(f"  freestream_ref ON: obs_mean[0] {old_um:.4f}→{u_inf:.4f}（全場背景→0, inflow 擾動→0）")
    u_inf_n = (u_inf - float(obs_mean[0])) / float(obs_std[0])
    u_zero_n = (0.0 - float(obs_mean[0])) / float(obs_std[0])
    v_zero_n = (0.0 - float(obs_mean[1])) / float(obs_std[1])
    if args.use_bc:
        print(f"  BC: u_inf={u_inf:.4f} (norm {u_inf_n:.3f}) w_bc={args.w_bc} bc_body_w={args.bc_body_w}")
    if args.hard_wall:
        print("  hard_wall ON: v=φ(y)·v_net ansatz（slip soft loss 停用，wall 強制 v=0）")
    if args.body_hard:
        print(f"  body_hard ON: u,v=clip(φ_body/{args.body_hard_scale},0,1)·net（body soft 停用，表面強制 u=v=0）")
    if args.brinkman:
        print(f"  brinkman ON: momentum += χ·u/η (η={args.brinkman_eta})（body mask 移除，body soft 停用）")
    # CfC 走時間軸：sensor_vals 需 [T,K,C]
    sv_TKC = jnp.transpose(sv, (1, 0, 2))
    RE_NORM = float(np.log(RE) / np.log(10000.0))
    T_total = float(st[-1] - st[0]); st0 = float(st[0])
    print(f"  K={K} T={T} C={C}  Lx={Lx:.4f} Ly={Ly:.4f} Re={RE:.0f}  body=({bc[0]:.3f},{bc[1]:.3f}) R={br:.3f}")

    model = build_model(args.mode, bc, br, args.attention_kind,
                        fourier_embed_dim=args.fourier_embed_dim, t_total=T_total)
    print(f"  T_total(model)={T_total:.2f}（串接 runtime 時間窗，非 CFG 1.0）")
    if args.fourier_embed_dim:
        print(f"  fourier_embed_dim={args.fourier_embed_dim} (RFF, 非週期, 修 x=0≡x=L fold)")
    rng = np.random.RandomState(args.seed)
    init_xy = jnp.asarray(rng.uniform(0, 1, (8, 2)), jnp.float32)
    init_t = jnp.asarray(rng.uniform(st0, st0 + T_total, (8,)), jnp.float32)
    params = model.init(jax.random.PRNGKey(args.seed), sv_TKC, sp, RE_NORM, st, init_xy, init_t)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  params={n_params:,}")

    LO = LiquidOperator

    # hard-wall ansatz：φ(y)=4y(1-y)（C∞, φ(0)=φ(1)=0, φ(0.5)=1）。
    def phi_wall(y):
        return 4.0 * y * (1.0 - y)

    def constrain_uv(u_raw, v_raw, x, y):
        """raw normalized (u,v) → 套 body hard gate + wall ansatz 後的 normalized (u,v)。

        body_hard：pi-lnn Sukumar 2022 gate=clip(φ_body/scale,0,1)，body 表面→0（u=v=0）、fluid→1。
        hard_wall：v 額外乘 φ(y) envelope，wall 強制 v=0。
        constrained = gate·raw + (1-gate)·zero_n（zero_n 為物理 0 的 normalized 值）。
        """
        u_c, v_c = u_raw, v_raw
        if args.body_hard:
            phi_b = jnp.sqrt((x - bc[0]) ** 2 + (y - bc[1]) ** 2 + 1e-12) - br  # body SDF
            g = jnp.clip(phi_b / args.body_hard_scale, 0.0, 1.0)
            u_c = g * u_c + (1.0 - g) * u_zero_n
            v_c = g * v_c + (1.0 - g) * v_zero_n
        if args.hard_wall:
            p = phi_wall(y)
            v_c = p * v_c + (1.0 - p) * v_zero_n
        return u_c, v_c

    # field_fn：decode → denorm u,v（p identity, 無監督）
    def field_fn(p_, h_, x, y, t, sp_, st_):
        out = model.apply(p_, jnp.array([[x, y]]), jnp.array([t]), h_, st_, sp_,
                          method=LO.decode_query)[0]
        u_n, v_n = constrain_uv(out[0], out[1], x, y)
        u = u_n * obs_std[0] + obs_mean[0]
        v = v_n * obs_std[1] + obs_mean[1]
        # concatenate 取代 stack（folx registry，避免 output full hessian fallback）
        return jnp.concatenate([u[None], v[None], out[2][None]])
    fused = make_fused_field_derivatives(field_fn)

    # sensor query 全量（重建目標）：(pos × time)
    xy_q_full = jnp.broadcast_to(sp[None], (T, K, 2)).reshape(T * K, 2)
    t_q_full = jnp.broadcast_to(st[:, None], (T, K)).reshape(T * K)
    tgt_full = jnp.transpose(sv, (1, 0, 2)).reshape(T * K, 2)   # normalized target

    def sdf(x, y):
        return jnp.sqrt((x - bc[0]) ** 2 + (y - bc[1]) ** 2 + 1e-8) - br

    def loss_fn(p_, al_state, cx, cy, ct, sq_idx, bc_in, bc_body, bc_slip):
        h = model.apply(p_, sv_TKC, sp, RE_NORM, st, method=LO.encode)
        # data loss（sensor mini-batch）；hard_wall 時 v 走 constrained ansatz
        xq, tq, tg = xy_q_full[sq_idx], t_q_full[sq_idx], tgt_full[sq_idx]

        def _decode_sumsq(xqc, tqc, tgc):
            pred = model.apply(p_, xqc, tqc, h, st, sp, method=LO.decode_query)
            pu, pv = constrain_uv(pred[:, 0], pred[:, 1], xqc[:, 0], xqc[:, 1])
            return jnp.sum((pu - tgc[:, 0]) ** 2 + (pv - tgc[:, 1]) ** 2)

        nq = xq.shape[0]
        if args.data_chunks <= 1:
            data_loss = _decode_sumsq(xq, tq, tg) / nq
        else:
            # decode 分塊 + remat：大 K 時 vector attention [n_q,K,hidden] 爆，分塊降峰值
            # 保總採樣量（公平）；remat 讓 backward 重算每塊（同 chunked fof）。
            DC = args.data_chunks
            cs = nq // DC                                  # n_sensor_q 須整除 data_chunks
            xr = xq[:cs * DC].reshape(DC, cs, 2)
            tr = tq[:cs * DC].reshape(DC, cs)
            gr = tg[:cs * DC].reshape(DC, cs, 2)
            sums = jax.lax.map(jax.checkpoint(lambda a: _decode_sumsq(*a)), (xr, tr, gr))
            data_loss = jnp.sum(sums) / (cs * DC)
        # v2 BC losses（比對 normalized 預測 vs normalized 目標）
        bc_loss = 0.0
        if args.use_bc:
            def dec(xyt):
                return model.apply(p_, xyt[:, :2], xyt[:, 2], h, st, sp, method=LO.decode_query)
            o_in = dec(bc_in)                                       # inflow x=0
            bc_loss = bc_loss + jnp.mean((o_in[:, 0] - u_inf_n) ** 2) + jnp.mean((o_in[:, 1] - v_zero_n) ** 2)
            if not (args.body_hard or args.brinkman):  # body 由 hard gate / Brinkman 強制時免 soft
                o_bd = dec(bc_body)                                 # body no-slip u=v=0
                bc_loss = bc_loss + args.bc_body_w * (
                    jnp.mean((o_bd[:, 0] - u_zero_n) ** 2) + jnp.mean((o_bd[:, 1] - v_zero_n) ** 2))
            if not args.hard_wall:   # hard_wall 時 slip v=0 由 ansatz 強制，免 soft loss
                o_sl = dec(bc_slip)                                 # slip walls v=0
                bc_loss = bc_loss + jnp.mean((o_sl[:, 1] - v_zero_n) ** 2)
        # physics（forcing=0, Lx/Ly 物理尺度化, body 遮罩）
        common = (sp, st)
        value, jac, d2x, d2y = fused(p_, h, cx, cy, ct, *common)
        u = value[:, 0]; v = value[:, 1]
        u_x = jac[:, 0, 0] / Lx; u_y = jac[:, 0, 1] / Ly; u_t = jac[:, 0, 2]
        v_x = jac[:, 1, 0] / Lx; v_y = jac[:, 1, 1] / Ly; v_t = jac[:, 1, 2]
        p_x = jac[:, 2, 0] / Lx; p_y = jac[:, 2, 1] / Ly
        u_xx = d2x[:, 0] / Lx ** 2; u_yy = d2y[:, 0] / Ly ** 2
        v_xx = d2x[:, 1] / Lx ** 2; v_yy = d2y[:, 1] / Ly ** 2
        mom_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        cont = u_x + v_y
        d_body = sdf(cx, cy)
        if args.brinkman:
            # 沉浸邊界：body 內外同方程，momentum 加 χ·u/η（χ=body indicator），body 內 u→0。
            # physics 全域算（不 mask body），讓固體連續嵌入場。
            chi = (d_body < 0.0).astype(jnp.float32)
            mom_u = mom_u + chi * u / args.brinkman_eta
            mom_v = mom_v + chi * v / args.brinkman_eta
            mask = jnp.ones_like(cx)
        else:
            mask = (d_body > 0.0).astype(jnp.float32)          # 只在流體域算 physics
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        cont_mse = jnp.sum(mask * cont ** 2) / denom
        mom = jnp.sum(mask * (mom_u ** 2 + mom_v ** 2)) / denom
        if args.use_al:
            # continuity 走 ALM：phys 只留 momentum，cont 由 λ·C+ρ/2·C² 強制
            phys = mom
            # al_loss_term 已於 2026-08-04 從 losses.py 移除——它零 production 呼叫者，
            # 且其 docstring 自承主線 pipeline 不可用它（signed_mean 的 λ 項與 ρ 項
            # 用不同的量，assembly.py 因此內聯重寫）。這裡是它唯一的呼叫端，
            # 公式原樣搬入：λ·C + (ρ/2)·C²。
            al_term = al_state.lambda_ * cont_mse + 0.5 * al_state.rho * cont_mse ** 2
        else:
            # 舊行為：continuity 與 momentum 等權併入 phys
            phys = mom + cont_mse
            al_term = 0.0
        total = data_loss + args.w_phys * phys + args.w_bc * bc_loss + al_term
        return total, (data_loss, mom, jnp.asarray(bc_loss), cont_mse)

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt = tx.init(params)

    @jax.jit
    def step(p_, o_, al_state, cx, cy, ct, sq_idx, bc_in, bc_body, bc_slip):
        # value_and_grad 只對 p_（argnum 0）求導；al_state 是 buffer-like（λ 無 grad）
        (L, (dl, mo, bl, cm)), g = jax.value_and_grad(loss_fn, has_aux=True)(
            p_, al_state, cx, cy, ct, sq_idx, bc_in, bc_body, bc_slip)
        upd, o_ = tx.update(g, o_, p_)
        return optax.apply_updates(p_, upd), o_, L, dl, mo, bl, cm

    # ── BC 取樣（固定 shape；use_bc=False 時用最小 dummy，loss 內不計）──
    nbc = args.bc_n if args.use_bc else 1

    def sample_bc(key):
        kk = jax.random.split(key, 6)
        # inflow x=0
        y_in = jax.random.uniform(kk[0], (nbc,), jnp.float32, 0.0, 1.0)
        t_in = jax.random.uniform(kk[1], (nbc,), jnp.float32, st0, st0 + T_total)
        bc_in = jnp.stack([jnp.zeros(nbc), y_in, t_in], -1)
        # body no-slip 在圓周表面（非填實心圓盤）：rr=br 固定半徑，避免稀釋表面取樣密度
        rr = jnp.full((nbc,), br, jnp.float32)
        th = jax.random.uniform(kk[3], (nbc,), jnp.float32, 0.0, 2 * jnp.pi)
        bx = bc[0] + rr * jnp.cos(th); by = bc[1] + rr * jnp.sin(th)
        t_bd = jax.random.uniform(kk[4], (nbc,), jnp.float32, st0, st0 + T_total)
        bc_body = jnp.stack([bx, by, t_bd], -1)
        # slip walls: half y=0 half y=1
        half = nbc // 2
        xs = jax.random.uniform(kk[5], (nbc,), jnp.float32, 0.0, 1.0)
        ys = jnp.concatenate([jnp.zeros(half), jnp.ones(nbc - half)])
        ts = jax.random.uniform(kk[0], (nbc,), jnp.float32, st0, st0 + T_total)
        bc_slip = jnp.stack([xs, ys, ts], -1)
        return bc_in.astype(jnp.float32), bc_body.astype(jnp.float32), bc_slip.astype(jnp.float32)

    # ── 訓練 ──
    # ALM state：use_al=False 時為 dummy（loss 走 else 分支不碰）。
    al_state = al_init(init_lambda=0.0, rho=args.al_rho,
                       lambda_clip=args.al_lambda_clip, ema_momentum=args.al_ema_momentum)
    if args.use_al:
        print(f"  ALM continuity: ρ={args.al_rho} λ_clip={args.al_lambda_clip} "
              f"freeze=[0,{args.al_freeze_start}) update∈[{args.al_freeze_start},"
              f"{args.al_freeze_end}) ×{args.al_update_freq} ema={args.al_ema_momentum}")
    rk = jax.random.PRNGKey(args.seed + 1)
    t0 = time.time(); last = t0
    print(f"\n{'step':>6s} {'total':>11s} {'data':>11s} {'mom':>11s} {'cont':>11s} {'bc':>11s} {'lam':>7s} {'wall':>7s}")
    for s in range(args.steps + 1):
        rk, k1, k2 = jax.random.split(rk, 3)
        ks = jax.random.split(k1, 4)
        cx = jax.random.uniform(ks[0], (args.n_collo,), jnp.float32, 0.0, 1.0)
        cy = jax.random.uniform(ks[1], (args.n_collo,), jnp.float32, 0.0, 1.0)
        ct = jax.random.uniform(ks[2], (args.n_collo,), jnp.float32, st0, st0 + T_total)
        sq_idx = jax.random.choice(ks[3], T * K, (args.n_sensor_q,), replace=False)
        bc_in, bc_body, bc_slip = sample_bc(k2)
        params, opt, L, dl, mo, bl, cm = step(params, opt, al_state, cx, cy, ct, sq_idx, bc_in, bc_body, bc_slip)
        # ALM dual update：僅在 [freeze_start, freeze_end) 視窗內、每 freq 步推進 λ
        if args.use_al and args.al_freeze_start <= s < args.al_freeze_end and s % args.al_update_freq == 0:
            al_state = al_update(al_state, cm)
        if s == 0 or (s + 1) % 500 == 0 or s == args.steps:
            wall = time.time() - last; last = time.time()
            print(f"{s:>6d} {float(L):>11.4e} {float(dl):>11.4e} {float(mo):>11.4e} "
                  f"{float(cm):>11.4e} {float(bl):>11.4e} {float(al_state.lambda_):>7.3f} {wall:>7.2f}")
        if not np.isfinite(float(L)):
            print(f"[FATAL] NaN at step {s}"); break
    print(f"train wall: {time.time()-t0:.1f}s")

    # ── DNS field-L2 eval ──
    dns_u = np.asarray(d["dns_u"]); dns_v = np.asarray(d["dns_v"])  # [n_eval,H,W] 物理
    dns_x = np.asarray(d["dns_x"]); dns_y = np.asarray(d["dns_y"])  # normalized grid
    dns_t_idx = np.asarray(d["dns_t_idx"])
    Hn, Wn = dns_u.shape[1], dns_u.shape[2]
    gx, gy = np.meshgrid(dns_x, dns_y)                              # [H,W]
    gx = jnp.asarray(gx.reshape(-1), jnp.float32); gy = jnp.asarray(gy.reshape(-1), jnp.float32)
    h_eval = model.apply(params, sv_TKC, sp, RE_NORM, st, method=LO.encode)

    xy_grid = jnp.stack([gx, gy], axis=-1)             # [H*W, 2]
    EVAL_CHUNK = 2048    # vector attention 中間張量 [3N,K,hidden] 隨 N 線性成長：
                         # N=32768→5GB/張量；chunk=2048→320MB，安全在 24GB VRAM 內。

    @jax.jit
    def recon_chunk(xy_chunk, tq_chunk):
        out = model.apply(params, xy_chunk, tq_chunk,
                          h_eval, st, sp, method=LO.decode_query)
        u_n, v_n = constrain_uv(out[:, 0], out[:, 1], xy_chunk[:, 0], xy_chunk[:, 1])  # body_hard/hard_wall 一致
        u = u_n * obs_std[0] + obs_mean[0]
        v = v_n * obs_std[1] + obs_mean[1]
        return u, v

    def recon(tq):
        n = xy_grid.shape[0]
        us, vs = [], []
        for s in range(0, n, EVAL_CHUNK):
            xc = xy_grid[s:s + EVAL_CHUNK]
            tc = jnp.full((xc.shape[0],), tq, dtype=jnp.float32)
            uc, vc = recon_chunk(xc, tc); jax.block_until_ready(uc)
            us.append(np.asarray(uc)); vs.append(np.asarray(vc))
        return np.concatenate(us), np.concatenate(vs)

    errs_u, errs_v = [], []
    ke_ratios, ke_relerrs, omega_rmses, div_l2s = [], [], [], []
    # body 內遮罩（流體域，所有判據共用）
    gx2 = np.asarray(gx).reshape(Hn, Wn); gy2 = np.asarray(gy).reshape(Hn, Wn)
    m = (np.sqrt((gx2 - bc[0]) ** 2 + (gy2 - bc[1]) ** 2) - br) > 0
    for i, ti in enumerate(dns_t_idx):
        tq = float(st[ti])
        pu, pv = recon(tq)
        pu = pu.reshape(Hn, Wn); pv = pv.reshape(Hn, Wn)
        du, dv = dns_u[i], dns_v[i]
        # field-L2（per-channel，輔助判據）
        eu = np.linalg.norm((pu - du) * m) / (np.linalg.norm(du * m) + 1e-8)
        ev = np.linalg.norm((pv - dv) * m) / (np.linalg.norm(dv * m) + 1e-8)
        errs_u.append(eu); errs_v.append(ev)
        # ── 主判據：KE / vorticity / divergence（§7：cylinder 判生死用 KE，不用 field-L2）──
        ke_p = 0.5 * (pu ** 2 + pv ** 2); ke_r = 0.5 * (du ** 2 + dv ** 2)
        ke_ratios.append(float(np.sum(ke_p * m) / (np.sum(ke_r * m) + 1e-8)))            # ke_pred/ke_ref
        ke_relerrs.append(float(np.linalg.norm((ke_p - ke_r) * m) / (np.linalg.norm(ke_r * m) + 1e-8)))
        # 物理導數：dns_x/dns_y 為 normalized 軸，x_phys=x_norm·Lx → ∂/∂x_phys=(1/Lx)·∂/∂x_norm
        w_p = np.gradient(pv, dns_x, axis=1) / Lx - np.gradient(pu, dns_y, axis=0) / Ly   # ω=∂v/∂x-∂u/∂y
        w_r = np.gradient(dv, dns_x, axis=1) / Lx - np.gradient(du, dns_y, axis=0) / Ly
        omega_rmses.append(float(np.sqrt(np.sum(((w_p - w_r) * m) ** 2) / (np.sum(m) + 1e-8))))
        div_p = np.gradient(pu, dns_x, axis=1) / Lx + np.gradient(pv, dns_y, axis=0) / Ly  # ∇·u（pred）
        div_l2s.append(float(np.linalg.norm(div_p * m)))
        if args.dump_fields and i == 0:   # 存第一個 val 時刻的場供畫圖
            np.savez_compressed(
                args.dump_fields,
                pred_u=pu, pred_v=pv, dns_u=du, dns_v=dv,
                pred_w=w_p, dns_w=w_r, gx=gx2, gy=gy2, mask=m.astype(np.float32),
                sensor_pos=np.asarray(sp), body_center=np.asarray(bc), body_radius=np.float32(br),
                u_rel=eu, v_rel=ev,
            )
            print(f"[dump_fields] 已存 {args.dump_fields}（t_idx={ti}, u_rel={eu:.3f} v_rel={ev:.3f}）")
    print(f"\n=== KE / vorticity (PRIMARY criterion, mode={args.mode}, {len(dns_t_idx)} val times) ===")
    print(f"  ke_pred/ke_ref = {np.mean(ke_ratios):.4f}   KE rel-err = {np.mean(ke_relerrs):.4f}")
    print(f"  omega RMSE     = {np.mean(omega_rmses):.4f}   div L2 = {np.mean(div_l2s):.4f}")
    print(f"\n=== DNS field-L2 (auxiliary, mode={args.mode}) ===")
    print(f"  u rel-L2 = {np.mean(errs_u):.4f}  v rel-L2 = {np.mean(errs_v):.4f}")


if __name__ == "__main__":
    main()

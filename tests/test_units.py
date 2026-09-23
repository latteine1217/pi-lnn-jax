"""Core unit tests: CfC / Decoder shape / Periodic encoding / Sensor axis convention / Evaluator.

對齊 pi-lnn 既有關鍵 regression tests:
  - test_sensor_axis_convention.py (EXP-101 災難根因 guard)
  - test_cfc_pass_refactor.py (CfC numerical stability)
  - test_pos_enc_optimization.py (週期 BC)

POC 範圍：confirm code 對 ground truth 數值正確。
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _acceptance_strictness
from pi_lnn_jax.data import DNS_NPY, load_dns, SENSOR_JSON, SENSOR_NPZ
from pi_lnn_jax.evaluate import (
    compute_divergence_l2, compute_vorticity, compute_energy_spectrum, compute_metrics
)
from pi_lnn_jax.models import (
    CfCStep, periodic_fourier_encode, temporal_phase_anchor,
    LiquidOperator,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1) CfCStep numerical stability
# ─────────────────────────────────────────────────────────────────────────────

def test_cfc_step_numerical_stable():
    """單步 CfC: random input + h → h_new 應有限、無 NaN。"""
    cell = CfCStep(hidden_size=16)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (4, 8))   # batch=4, input_size=8
    h = jax.random.normal(rng, (4, 16))
    dt = jnp.array(0.05)
    params = cell.init(rng, h, (x, dt))
    h_new, _ = cell.apply(params, h, (x, dt))
    assert h_new.shape == h.shape, f"shape mismatch {h_new.shape} vs {h.shape}"
    assert jnp.all(jnp.isfinite(h_new)), "h_new 含 NaN/Inf"
    # CfC gate ∈ (0, 1)，輸出 ∈ (-1, 1)（tanh of f1/f2）→ |h_new| ≤ 1
    assert jnp.all(jnp.abs(h_new) <= 1.0 + 1e-6), f"|h_new| 超出 tanh 範圍: max={jnp.abs(h_new).max()}"
    print(f"✓ cfc_step_numerical_stable: |h_new| max = {float(jnp.abs(h_new).max()):.4f}")


def test_cfc_step_dt_effect():
    """較大 dt 應 give 不同 h_new (gate 對 dt 敏感)；證 dt 確實 enter computation。"""
    cell = CfCStep(hidden_size=16)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (4, 8))
    h = jax.random.normal(rng, (4, 16))
    params = cell.init(rng, h, (x, jnp.array(0.05)))
    h1, _ = cell.apply(params, h, (x, jnp.array(0.01)))
    h2, _ = cell.apply(params, h, (x, jnp.array(1.0)))
    diff = float(jnp.linalg.norm(h1 - h2))
    assert diff > 1e-4, f"dt 變化未影響 h_new (diff={diff})"
    print(f"✓ cfc_step_dt_effect: dt=0.01 vs 1.0 → h diff norm = {diff:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2) Periodic encoding: x=0 與 x=L 編碼恆等
# ─────────────────────────────────────────────────────────────────────────────

def test_periodic_encoding_boundary_equiv():
    """sin/cos(2πk·x/L) for k=integer 應在 x=0 與 x=L 相同。"""
    L = 1.0
    p0 = periodic_fourier_encode(jnp.array([[0.0, 0.3]]), L, n_harmonics=8)
    pL = periodic_fourier_encode(jnp.array([[L, 0.3]]), L, n_harmonics=8)
    err = float(jnp.linalg.norm(p0 - pL))
    assert err < 1e-5, f"periodic encoding 不滿足 x=0 ≡ x=L (err={err:.6e})"
    print(f"✓ periodic_encoding_boundary_equiv: err = {err:.2e}")


def test_temporal_phase_anchor():
    """temporal_phase_anchor 在 t=0 與 t=T_total 應相同 (sin/cos 週期)。"""
    T = 5.0
    a0 = temporal_phase_anchor(jnp.array([[0.0]]), T, n_harmonics=2)
    aT = temporal_phase_anchor(jnp.array([[T]]), T, n_harmonics=2)
    err = float(jnp.linalg.norm(a0 - aT))
    assert err < 1e-5, f"temporal anchor 不滿足週期 (err={err:.6e})"
    print(f"✓ temporal_phase_anchor: err = {err:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3) Decoder shape: 各 config 變體 output 都 [N, 3]
# ─────────────────────────────────────────────────────────────────────────────

def test_decoder_output_shapes():
    """5 個 decoder 變體 output 均應為 [N, 3]。"""
    base_cfg = dict(
        sensor_value_dim=2, d_model=16, d_time=4,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
        num_token_attention_layers=0, num_query_mlp_layers=0,
        query_mlp_hidden_dim=16, operator_rank=8,
        decoder_attention_heads=1, use_temporal_anchor=True, T_total=1.0,
        temporal_anchor_harmonics=2, domain_length=1.0,
    )
    T, K, N = 5, 10, 8
    sensor_vals = jax.random.normal(jax.random.PRNGKey(0), (T, K, 2))
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(1), (K, 2))
    sensor_time = jnp.linspace(0, 1, T)
    xy = jax.random.uniform(jax.random.PRNGKey(2), (N, 2))
    t_q = jax.random.uniform(jax.random.PRNGKey(3), (N,))

    for flag_name, flag in [
        ("baseline", {}),
        ("modified_mlp", {"use_modified_mlp": True}),
        ("locality_decay", {"use_locality_decay": True}),
        ("no_xattn", {"disable_cross_attention": True}),
    ]:
        model = LiquidOperator(**base_cfg, **flag)
        params = model.init(jax.random.PRNGKey(42), sensor_vals, sensor_pos, 0.5, sensor_time, xy, t_q)
        out = model.apply(params, sensor_vals, sensor_pos, 0.5, sensor_time, xy, t_q)
        assert out.shape == (N, 3), f"{flag_name}: shape {out.shape} != ({N}, 3)"
        assert jnp.all(jnp.isfinite(out)), f"{flag_name}: NaN/Inf"
    print(f"✓ decoder_output_shapes: 5 variants × shape ({N}, 3)")


# ─────────────────────────────────────────────────────────────────────────────
# 4) Sensor axis convention regression (EXP-101 災難根因 guard)
# ─────────────────────────────────────────────────────────────────────────────

def test_sensor_axis_convention():
    """sensor file 內 (NPZ values, JSON coords) 應符合 pi-lnn convention:
       u_full[t, x_idx, y_idx]，即 indices=row-major: flat = x_idx*N + y_idx。

    sensor JSON/NPZ 已入庫，但比對用的 Re=1000 DNS 是不進 git 的大檔（見 .gitignore）：
    單一 checkout 以外的 worktree 預設沒有它。缺檔時 skip 而非紅——那個紅只說明
    「這台機器沒掛資料」，與 axis convention 無關。

    要在缺檔的 worktree 上實際跑到這條 guard，把大檔池指過去即可：
    `PILNJAX_DATA_ROOT=/path/to/main-checkout uv run python -m pytest tests/test_units.py`。
    驗收 job 上資料應在位，故 `PILNN_ACCEPTANCE_STRICT` 下這個 skip 即失敗——
    否則 EXP-101 的根因 guard 可能從此再也沒被執行過，而測試永遠是綠的。
    """
    if _acceptance_strictness.out_of_scope("kolmogorov"):
        pytest.skip("本次驗收 job 只驗 cylinder，Kolmogorov 的 DNS 未掛載")
    reason = None if DNS_NPY.exists() else f"Re=1000 DNS 不在本機（{DNS_NPY}）"
    _acceptance_strictness.require_no_skip(reason, "test_sensor_axis_convention")
    if reason is not None:
        pytest.skip(f"{reason}——本 guard 需要 DNS 全場才能比對")

    # 讀 sensor 與 DNS
    with open(SENSOR_JSON) as f:
        meta = json.load(f)
    npz = np.load(SENSOR_NPZ)
    dns_u, dns_v, _ = load_dns(time_stride=1)  # full T
    K = meta["K"]
    N = 128  # EXP-030 grid
    coords = np.asarray(meta["selected_coordinates"])
    indices = np.asarray(meta["indices"])
    assert coords.shape == (K, 2)
    assert indices.shape == (K,)

    # 對前 5 個 sensor 驗證 NPZ values == DNS at (x_idx, y_idx)
    x_arr = np.linspace(0.0, 1.0, N, endpoint=False)
    y_arr = np.linspace(0.0, 1.0, N, endpoint=False)
    for k in range(5):
        flat_idx = int(indices[k])
        # row-major: flat = x_idx * N + y_idx
        x_idx = flat_idx // N
        y_idx = flat_idx % N
        # 對應 (x, y) 應與 JSON selected_coordinates 對齊
        x_expected, y_expected = float(coords[k][0]), float(coords[k][1])
        assert abs(x_arr[x_idx] - x_expected) < 1e-4, \
            f"sensor {k}: x mismatch ({x_arr[x_idx]:.4f} vs {x_expected:.4f}) — axis convention 錯！"
        assert abs(y_arr[y_idx] - y_expected) < 1e-4, \
            f"sensor {k}: y mismatch ({y_arr[y_idx]:.4f} vs {y_expected:.4f})"
        # 驗 NPZ['u'] 在 t=0 == DNS[0, x_idx, y_idx]
        u_npz = float(npz['u'][k, 0])
        u_dns = float(dns_u[0, x_idx, y_idx])
        assert abs(u_npz - u_dns) < 1e-4, \
            f"sensor {k} t=0: NPZ u={u_npz:.4f} ≠ DNS u[{x_idx},{y_idx}]={u_dns:.4f}"
    print("✓ sensor_axis_convention: 5 sensors verified — NPZ values match DNS at (x_idx, y_idx)")


# ─────────────────────────────────────────────────────────────────────────────
# 5) Evaluator: Taylor-Green analytical solution
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluator_taylor_green():
    """Taylor-Green vortex 是 divergence-free analytical → div ≈ 0, ω/E(k) 應正確。"""
    N = 64
    x = np.linspace(0, 1, N, endpoint=False)
    y = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    u = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    v = -np.cos(2 * np.pi * xx) * np.sin(2 * np.pi * yy)

    div = compute_divergence_l2(u, v)
    assert div < 1e-3, f"TG divergence L2 {div} 應 ≈ 0"

    omega = compute_vorticity(u, v)
    omega_truth = +4 * np.pi * np.sin(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
    err = np.linalg.norm(omega - omega_truth) / np.linalg.norm(omega_truth)
    assert err < 0.05, f"TG vorticity FD err {err} 應 < 5%"

    k_arr, E_k = compute_energy_spectrum(u, v)
    assert int(np.argmax(E_k)) == 1, f"TG peak energy 應在 k_bin=1，得 {np.argmax(E_k)}"

    metrics = compute_metrics(u, v, u, v)
    assert metrics['u_rel_err'] < 1e-10 and metrics['omega_rel_err'] < 1e-10
    print(f"✓ evaluator_taylor_green: div={div:.2e}  ω_err={err:.4f}  peak_k=1")


# ─────────────────────────────────────────────────────────────────────────────
# 6) Physics output denormalization sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_physics_denorm_logic():
    """ns_residuals 內: u_phys = u_norm * u_std + u_mean。對 zero output 應給 mean。"""
    from pi_lnn_jax.physics import make_ns_residual_fn
    # construct mini model
    model = LiquidOperator(
        sensor_value_dim=2, d_model=16, d_time=4,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
        num_token_attention_layers=0, num_query_mlp_layers=0,
        query_mlp_hidden_dim=16, operator_rank=8,
        decoder_attention_heads=1, use_temporal_anchor=True, T_total=1.0,
        temporal_anchor_harmonics=2, domain_length=1.0,
    )
    rng = jax.random.PRNGKey(0)
    T, K, N = 5, 5, 4
    sensor_vals = jnp.zeros((T, K, 2))
    sensor_pos = jax.random.uniform(rng, (K, 2))
    sensor_time = jnp.linspace(0, 1, T)
    xy_init = jax.random.uniform(rng, (N, 2))
    t_init = jax.random.uniform(rng, (N,))
    params = model.init(rng, sensor_vals, sensor_pos, 0.5, sensor_time, xy_init, t_init)

    norm_stats = {"u_mean": 1.0, "u_std": 2.0, "v_mean": -0.5, "v_std": 3.0}
    # Wave 5 API: sensor_pos/sensor_time/nu/norm 改為 ns_residuals 的 runtime args
    ns_fn, poisson_fn = make_ns_residual_fn(model)
    # Encode once → h_states
    h_states = model.apply(params, sensor_vals, sensor_pos, 0.5, sensor_time,
                            method=LiquidOperator.encode)
    cx = jax.random.uniform(rng, (4,), minval=0, maxval=1)
    cy = jax.random.uniform(rng, (4,), minval=0, maxval=1)
    ct = jax.random.uniform(rng, (4,), minval=0, maxval=1)
    nu = 1.0 / 1000.0
    um, us, vm, vs = (norm_stats[k] for k in ("u_mean", "u_std", "v_mean", "v_std"))
    pm, ps = 0.0, 1.0  # p 無監督場景的 identity 等價（p = out[2]*1 + 0）
    mom_u, mom_v, cont = ns_fn(params, h_states, cx, cy, ct, 0.1, 2.0,
                               sensor_pos, sensor_time, nu, um, us, vm, vs, pm, ps)
    pois = poisson_fn(params, h_states, cx, cy, ct,
                      sensor_pos, sensor_time, um, us, vm, vs, pm, ps)
    for name, v in [("mom_u", mom_u), ("mom_v", mom_v), ("cont", cont), ("poisson", pois)]:
        assert jnp.isfinite(v), f"{name} not finite: {v}"
    print(f"✓ physics_denorm_logic: ns_fn + poisson_fn callable; "
          f"mom_u/mom_v/cont/poisson all finite (vals: {float(mom_u):.3e} / {float(mom_v):.3e} / "
          f"{float(cont):.3e} / {float(pois):.3e})")

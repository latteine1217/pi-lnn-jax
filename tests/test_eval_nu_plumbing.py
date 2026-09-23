"""釘住 ν 從 eval script → evaluate_time_series → compute_metrics 的透傳。

band/γ 欄位需要 ν（k_η 邊界）。eval script 需 checkpoint 才能端對端驗，故此處用
stub model 驗證參數鏈路本身：傳了 ν 就要有 band 欄位，沒傳就必須缺席（不得靜默
退回固定邊界）。
"""
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.evaluate import evaluate_time_series


class _StubModel:
    """回傳與 query 座標相關的平滑場，確保譜非退化（否則 k_η 無意義）。"""

    def apply(self, params, sv, sp, rn, st, xy, t):
        u = jnp.sin(2 * jnp.pi * 3 * xy[:, 0]) * 0.5
        v = jnp.cos(2 * jnp.pi * 2 * xy[:, 1]) * 0.5
        return jnp.stack([u, v, jnp.zeros_like(u)], axis=-1)


def _fixture(N=32, T=3):
    rng = np.random.default_rng(0)
    xs = np.arange(N) / N
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dns_u = np.stack([np.sin(2 * np.pi * 3 * X) + 0.1 * rng.standard_normal((N, N))
                      for _ in range(T)])
    dns_v = np.stack([np.cos(2 * np.pi * 2 * Y) + 0.1 * rng.standard_normal((N, N))
                      for _ in range(T)])
    return dict(
        model=_StubModel(), params={}, sensor_vals=jnp.zeros((T, 4, 2)),
        sensor_pos=jnp.zeros((4, 2)), re_norm=1.0, sensor_time=jnp.arange(T, dtype=float),
        norm_stats={"u_mean": 0.0, "u_std": 1.0, "v_mean": 0.0, "v_std": 1.0},
        dns_u=dns_u, dns_v=dns_v, dns_t=np.arange(T, dtype=float), verbose=False,
    )


def test_nu_reaches_compute_metrics():
    out = evaluate_time_series(**_fixture(), nu=1e-4)
    mm = out["metrics_mean"]
    for key in ("uv_rel_err", "band_rel_err_low", "band_rel_err_mid",
                "gamma_low", "gamma_mid"):
        assert key in mm, f"傳了 ν 卻缺 {key}"
    assert np.isfinite(mm["band_rel_err_low"])
    assert "k_eta" in out["metrics_per_t"][0]


def test_without_nu_band_fields_absent():
    out = evaluate_time_series(**_fixture())
    mm = out["metrics_mean"]
    assert "uv_rel_err" in mm, "uv_rel_err 不依賴 ν，應永遠存在"
    for key in ("band_rel_err_low", "gamma_low"):
        assert key not in mm, f"未傳 ν 卻產生 {key}（不得靜默退回固定邊界）"


def test_p90_present_and_bounds_mean():
    """temporal p90 必須存在，且對非負指標不小於 mean（sanity）。"""
    out = evaluate_time_series(**_fixture(), nu=1e-4)
    assert "metrics_p90" in out
    assert out["metrics_p90"]["uv_rel_err"] >= out["metrics_mean"]["uv_rel_err"] - 1e-12


def test_band_high_diag_reports_validity():
    """high band 不進 headline，但必須顯性報告有效幀數與 k_cut 範圍。"""
    out = evaluate_time_series(**_fixture(), nu=1e-4)
    d = out["band_high_diag"]
    assert d is not None
    assert d["n_total_frames"] == 3
    assert 0 <= d["n_valid_frames"] <= 3
    assert d["k_cut_min"] <= d["k_cut_max"]
    assert "band_rel_err_high" not in out["metrics_mean"], "high band 不應進 headline"

"""`scripts/plot_extrapolation_leadtime.py` 的判讀量。

這張圖的兩條參考線決定「外推到底有沒有用」的結論，所以它們的定義要能在本機驗：
persistence 必須是「凍結資料末端的真值場」，去相關水平必須來自實際的遠距幀對，
而不是任何寫死的經驗值。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from plot_extrapolation_leadtime import (  # noqa: E402
    _first_crossing,
    _uv_rel,
    decorrelation_level,
    persistence_curve,
    useful_horizon,
)


def _fake_dns(n=41, N=8, seed=0):
    """時間上緩慢漂移 + 逐漸失相關的合成場（只需結構正確，不需物理正確）。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, n)
    base = rng.normal(size=(N, N))
    drift = rng.normal(size=(n, N, N))
    u = base[None] + 0.3 * np.cumsum(drift, axis=0) / np.sqrt(n)
    v = -base[None] + 0.3 * np.cumsum(drift[::-1], axis=0) / np.sqrt(n)
    return u, v, t


def test_persistence_is_zero_at_the_freeze_point():
    u, v, t = _fake_dns()
    t_out, err = persistence_curve(u, v, t, 5.0)

    i = int(np.argmin(np.abs(t_out - 5.0)))
    assert err[i] == pytest.approx(0.0, abs=1e-12), "凍結點自己對自己必須是 0"
    assert err[i + 1] > 0.0 and err[-1] > err[i + 1], "離凍結點越遠應該越差"
    print("✓ persistence_is_zero_at_the_freeze_point")


def test_persistence_refuses_off_grid_freeze_time():
    """時間軸上沒有那個時刻就 raise，不取最近幀了事。"""
    u, v, t = _fake_dns()
    with pytest.raises(ValueError, match="沒有 t="):
        persistence_curve(u, v, t, 5.123)
    print("✓ persistence_refuses_off_grid_freeze_time")


def test_decorrelation_level_uses_far_apart_pairs():
    u, v, t = _fake_dns()
    mean, std, n_pairs = decorrelation_level(u, v, t, lag=4.0)

    assert n_pairs > 0 and mean > 0.0 and std >= 0.0
    # 鑑別力自檢：遠距幀對的誤差必須明顯大於相鄰幀
    adjacent = _uv_rel(u[0], v[0], u[1], v[1])
    assert mean > adjacent
    print("✓ decorrelation_level_uses_far_apart_pairs")


def test_decorrelation_level_raises_when_span_too_short():
    u, v, t = _fake_dns()
    with pytest.raises(ValueError, match="時間跨度不足"):
        decorrelation_level(u, v, t, lag=100.0)
    print("✓ decorrelation_level_raises_when_span_too_short")


def test_first_crossing_semantics():
    tau = np.array([0.0, 1.0, 2.0, 3.0])
    assert _first_crossing(tau, np.array([0.1, 0.2, 0.6, 0.9]), 0.5) == 2.0
    assert _first_crossing(tau, np.array([0.1, 0.2, 0.3, 0.4]), 0.5) is None
    # level 可為逐點陣列（persistence 曲線）
    assert _first_crossing(tau, np.array([0.1, 0.9, 0.9, 0.9]),
                           np.array([0.5, 0.5, 2.0, 2.0])) == 1.0
    print("✓ first_crossing_semantics")


def test_useful_horizon_ignores_the_tau_zero_artifact():
    """persistence 在 τ=0 依定義為 0，不能讓那個點決定水平線。"""
    tau = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    pers = np.array([0.0, 0.4, 0.8, 1.0, 1.1])

    # 模型在 τ=0.5/1.0 較好、1.5 起輸 → 水平線 = 1.0（τ=0 的定義性交叉不算）
    model = np.array([0.14, 0.3, 0.6, 1.05, 1.2])
    assert useful_horizon(tau, model, pers) == 1.0
    # 一開始就輸 → 0.0（外推不如把場凍住）
    assert useful_horizon(tau, np.array([0.14, 0.5, 0.9, 1.1, 1.3]), pers) == 0.0
    # 全程都贏 → 最大 τ
    assert useful_horizon(tau, np.full(5, 0.05), pers) == 2.0
    print("✓ useful_horizon_ignores_the_tau_zero_artifact")


def test_uncorrelated_reference_is_derived_not_measured():
    """√2 是推導值：兩個等能量、獨立、零均值的場之間的相對 L2。"""
    from plot_extrapolation_leadtime import UNCORRELATED_SQRT2

    rng = np.random.default_rng(7)
    a = rng.normal(size=(64, 64))
    b = rng.normal(size=(64, 64))          # 與 a 獨立、同能量
    au, av = a, rng.normal(size=(64, 64))
    bu, bv = b, rng.normal(size=(64, 64))

    empirical = _uv_rel(au, av, bu, bv)
    assert abs(empirical - UNCORRELATED_SQRT2) < 0.05, (
        f"獨立等能量場的相對 L2 應趨近 √2，得 {empirical}")
    print("✓ uncorrelated_reference_is_derived_not_measured")


def test_cli_end_to_end(tmp_path):
    """合成 series.npz + DNS .npy 走完整條 CLI：欄位契約與輸出檔一次驗完。"""
    u, v, t = _fake_dns()
    dns_path = tmp_path / "dns.npy"
    np.save(dns_path, {"u": u, "v": v, "time": t,
                       "x": np.arange(u.shape[1]), "y": np.arange(u.shape[2])},
            allow_pickle=True)

    err = np.linspace(0.05, 1.2, len(t))
    series_path = tmp_path / "series.npz"
    np.savez(series_path, t=t, uv_rel_err=err)

    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/plot_extrapolation_leadtime.py"),
         "--series", str(series_path), "--dns", str(dns_path), "--t-data-end", "5.0"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert rc.returncode == 0, rc.stderr[-2000:]
    import json
    report = json.loads((tmp_path / "extrapolation_leadtime.json").read_text())
    assert report["t_data_end"] == 5.0
    assert set(report["at_lead_times"]) == {"0", "0.5", "1", "2", "5"}
    assert set(report["at_lead_times"]["1"]) == {"model", "persistence_oracle"}
    assert report["decorrelation_level"]["n_pairs"] > 0
    assert report["useful_horizon_vs_persistence_model"] is None  # 沒給 fields
    assert (tmp_path / "extrapolation_leadtime.pdf").exists()
    print("✓ cli_end_to_end")


def test_cli_with_model_persistence(tmp_path):
    """給了 fields.npz 就要多出 model-persistence 那條線與水平線。"""
    u, v, t = _fake_dns()
    dns_path = tmp_path / "dns.npy"
    np.save(dns_path, {"u": u, "v": v, "time": t,
                       "x": np.arange(u.shape[1]), "y": np.arange(u.shape[2])},
            allow_pickle=True)
    series_path = tmp_path / "series.npz"
    np.savez(series_path, t=t, uv_rel_err=np.linspace(0.05, 1.2, len(t)))
    fields_path = tmp_path / "fields.npz"
    rng = np.random.default_rng(1)
    np.savez(fields_path, t=t,
             u_pred=(u + 0.1 * rng.normal(size=u.shape)).astype(np.float32),
             v_pred=(v + 0.1 * rng.normal(size=v.shape)).astype(np.float32))

    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/plot_extrapolation_leadtime.py"),
         "--series", str(series_path), "--dns", str(dns_path),
         "--fields", str(fields_path), "--t-data-end", "5.0"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert rc.returncode == 0, rc.stderr[-2000:]
    import json
    report = json.loads((tmp_path / "extrapolation_leadtime.json").read_text())
    assert "persistence_model" in report["at_lead_times"]["1"]
    assert report["useful_horizon_vs_persistence_model"] is not None
    print("✓ cli_with_model_persistence")

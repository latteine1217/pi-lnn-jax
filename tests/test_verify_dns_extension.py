"""`scripts/verify_dns_extension.py` 的判定邏輯。

這支腳本的判定會決定「延長版 DNS 能不能當 eval 真值」，錯判的代價是整個外推
實驗的數字失去意義——所以判定函式要能在本機驗，不是只有 lab-server 跑得到。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from verify_dns_extension import compare  # noqa: E402

CFG = {"N": 256, "L": 1.0, "nu": 1e-4, "A": 0.1, "k_f": 2, "dt": 2.5e-4,
       "save_interval": 100, "integrator": "etdrk4", "dealias_mode": "3/2",
       "seed": 42, "ic_mode": "band_limited_random",
       "downsample_stride": 4, "source_N": 1024}


def _dns(n_frames, *, noise=0.0, backend="numpy", cfg_override=None, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_frames, 4, 4))
    cfg = dict(CFG, backend=backend, T_end=0.025 * (n_frames - 1))
    cfg.update(cfg_override or {})
    return {
        "time": np.arange(n_frames) * 0.025,
        "u": base + noise * rng.normal(size=base.shape),
        "v": 2.0 * base + noise * rng.normal(size=base.shape),
        "config": cfg,
    }


def test_identical_trajectory_passes():
    ref = _dns(11)
    ext = _dns(21)
    ext["u"][:11], ext["v"][:11] = ref["u"], ref["v"]
    report = compare(ext, ref, rtol=1e-6)

    assert report["verdict"] == "PASS"
    assert report["max_rel_l2"]["u"] == 0.0
    assert report["shared_frames"] == 11
    assert all(v is None for v in report["error_growth_crossings"].values())
    print("✓ identical_trajectory_passes")


def test_diverged_trajectory_fails_and_reports_growth():
    ref = _dns(11)
    ext = _dns(21)
    # 共同窗前段一致、後段指數發散：正是混沌放大要抓的形狀
    ext["u"][:11], ext["v"][:11] = ref["u"].copy(), ref["v"].copy()
    growth = np.exp(np.arange(11) - 6.0)[:, None, None]
    ext["u"][:11] += growth * np.abs(ref["u"]) * 1e-3
    ext["v"][:11] += growth * np.abs(ref["v"]) * 1e-3

    report = compare(ext, ref, rtol=1e-6)
    assert report["verdict"] == "FAIL"
    assert report["error_growth_crossings"]["t_first_exceeds_1e-08"] is not None
    assert report["max_rel_l2"]["u"] > 1e-6
    print("✓ diverged_trajectory_fails_and_reports_growth")


def test_backend_difference_alone_does_not_block_comparison():
    """backend 不同正是要量的東西，不能拿它當拒絕比對的理由。"""
    ref = _dns(11, backend="numpy")
    ext = _dns(21, backend="torch-cuda")
    ext["u"][:11], ext["v"][:11] = ref["u"], ref["v"]
    assert compare(ext, ref, rtol=1e-6)["verdict"] == "PASS"
    print("✓ backend_difference_alone_does_not_block_comparison")


def test_legacy_ic_mode_alias_is_accepted():
    """2026-04 產的參考檔只有 `init_mode`；別名是同一個量，不得判成設定不同。"""
    ref = _dns(11)
    ref["config"]["init_mode"] = ref["config"].pop("ic_mode")
    ext = _dns(21)                      # 新檔寫 ic_mode
    ext["u"][:11], ext["v"][:11] = ref["u"], ref["v"]

    assert compare(ext, ref, rtol=1e-6)["verdict"] == "PASS"
    # 鑑別力自檢：別名查得到不代表值不比——值真的不同時仍要炸
    ref["config"]["init_mode"] = "spectral_seeded"
    with pytest.raises(ValueError, match="數值設定不同"):
        compare(ext, ref, rtol=1e-6)
    print("✓ legacy_ic_mode_alias_is_accepted")


def test_numerical_setting_mismatch_raises():
    ref = _dns(11)
    ext = _dns(21, cfg_override={"dt": 1.95e-4})
    with pytest.raises(ValueError, match="數值設定不同"):
        compare(ext, ref, rtol=1e-6)
    print("✓ numerical_setting_mismatch_raises")


def test_misaligned_time_axis_raises():
    """時間軸不同就拒絕，不做寬鬆對齊（eval 紀律）。"""
    ref = _dns(11)
    ext = _dns(21)
    ext["time"] = ext["time"] + 0.001
    with pytest.raises(ValueError, match="時間軸不相同"):
        compare(ext, ref, rtol=1e-6)
    print("✓ misaligned_time_axis_raises")


def test_extended_shorter_than_reference_raises():
    with pytest.raises(ValueError, match="短於參考版"):
        compare(_dns(5), _dns(11), rtol=1e-6)
    print("✓ extended_shorter_than_reference_raises")

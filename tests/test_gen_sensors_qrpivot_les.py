"""`gen_sensors_qrpivot_les` 的渦度軸向與端到端輸出契約。

舊 LES 生成器把 [T, x, y] 當 [T, y, x] 微分，"omega" 變成 ∂v/∂y − ∂u/∂x 且不報錯。
這裡用解析的無散度場釘住：特徵裡的 omega 是真的旋度（不是對調軸向的那個量），
以及寫出的座標與 DNS 取值一致、各預算巢狀。
"""
from __future__ import annotations

import json
import sys

import numpy as np

from pi_lnn_jax.data import load_sensors_from_path
from scripts.gen_sensors_qrpivot_les import FEATURES, les_features
from tests.test_gen_sensors_qrpivot_dns import L, _streamfunction_field


def test_omega_feature_is_curl_not_axis_swapped():
    u, v, omega = _streamfunction_field(128)
    f = les_features(u, v, L)
    rel = np.linalg.norm(f["omega"] - omega) / np.linalg.norm(omega)
    assert rel < 5e-3  # 二階中央差分的截斷誤差
    # 舊生成器的量：∂v/∂y − ∂u/∂x，與真旋度明顯不同
    h = L / u.shape[-1]
    legacy = ((np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1))
              - (np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2))) / (2 * h)
    assert np.linalg.norm(legacy - omega) / np.linalg.norm(omega) > 0.5
    assert tuple(f) == FEATURES


def test_end_to_end_coordinates_values_and_nesting(tmp_path, monkeypatch):
    N, T = 32, 12
    u, v, _ = _streamfunction_field(N, T)
    t = np.linspace(0.0, 1.1, T)
    x = np.arange(N) * L / N
    les = {"time": t + 20.0, "u": u.astype(np.float32), "v": v.astype(np.float32),
           "config": {"L": L, "N": N}}
    rng = np.random.default_rng(0)
    du, dv = u * 0.9 + 0.01 * rng.standard_normal(u.shape), v * 0.9
    dns = {"time": t, "x": x, "y": x, "u": du, "v": dv}
    np.save(tmp_path / "les.npy", les, allow_pickle=True)
    np.save(tmp_path / "dns.npy", dns, allow_pickle=True)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "gen", "--les", str(tmp_path / "les.npy"), "--dns", str(tmp_path / "dns.npy"),
        "--budgets", "5", "10", "--t-spinup", "20.0", "--out-dir", str(out), "--tag", "syn"])
    from scripts.gen_sensors_qrpivot_les import main
    assert main() == 0

    metas = {K: json.loads((out / f"sensors_qrpivot_K{K}_syn.json").read_text()) for K in (5, 10)}
    c5, c10 = (np.asarray(metas[K]["selected_coordinates"]) for K in (5, 10))
    np.testing.assert_array_equal(c10[:5], c5)  # 巢狀
    vals = np.load(out / "sensors_qrpivot_K10_syn_dns_values.npz")
    ix, iy = np.rint(c10[:, 0] * N / L).astype(int), np.rint(c10[:, 1] * N / L).astype(int)
    np.testing.assert_allclose(vals["u"], du[:, ix, iy].T.astype(np.float32))
    np.testing.assert_allclose(vals["v"], dv[:, ix, iy].T.astype(np.float32))
    assert metas[10]["features"] == list(FEATURES)
    loaded = load_sensors_from_path(str(out / "sensors_qrpivot_K10_syn.json"))
    assert loaded is not None

#!/usr/bin/env python
"""tab:fair classical baselines — RBF multiquadric / IDW / div-free trig LSQ。

從 K=100 LES 感測器逐幀空間重建全場，與 PI-CON 用同一份 sensor_json/DNS/評估協定
（evaluation_protocol.load_for_evaluation）＋同一道 metric seam
（metric_artifact.evaluate_field_series）→ fair by construction。

⚠️ 2026-08 起 --protocol 必填且 --sensor-time-stride 不再預設 1：取樣由協定決定。
   既有 tab_fair_* 產物是以 stride 1 / T 200 產生的，與此不同——見
   docs/adr/0003（若已建立）與 knowledge/codebase/technical-debt.md 的 TD-4。

超參對齊論文：RBF ε=10、IDW p=2、div-free trig k_max=5。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pi_lnn_jax.evaluation_protocol import ProtocolMode

#: tab:fair 既有的每-method 欄位順序。投影必須逐鍵逐序與舊版相同——
#: 這張表進論文，欄位漂移比數字漂移更難察覺。
_AGG_KEYS_BASE = ["uv_rel_err", "u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err",
                  "ke_pw_mape", "ke_pw_nmae"]
_AGG_KEYS_BANDED = ["band_rel_err_low", "band_rel_err_mid", "gamma_low", "gamma_mid"]


def method_agg(projection: dict, *, banded: bool) -> dict:
    """把一筆 metric artifact 的投影塌成 tab:fair 的單列。

    值全部取自 `aggregate_metric_rows` 的輸出——與 artifact 同一次聚合，
    故兩者不可能各自漂移。`ke_t_mape` 的 pointwise 語意與 `ke_mape_def`
    標記由該函式寫入，不再由本腳本手寫。
    """
    mean = projection["metrics_mean"]
    ket = projection["ke_t_errors"]
    keys = _AGG_KEYS_BASE + (_AGG_KEYS_BANDED if banded else [])
    # 無定義的欄位落 null 而非 NaN：格點相對 k_eta 太粗時 band/γ 逐幀無定義，
    # 而 json.dump 會把 NaN 寫成字面 NaN——那不是合法 JSON，且下游印出來
    # 像個數字。null 讓消費端在索引時大聲失敗。
    agg = {k: (float(mean[k]) if np.isfinite(mean[k]) else None) for k in keys}
    agg["ke_t_mape"] = float(ket["ke_t_mape"])
    agg["ke_t_mape_spatialmean"] = float(ket["ke_t_mape_spatialmean"])
    agg["ke_t_rel_l2"] = float(ket["ke_t_rel_l2"])
    agg["ke_mape_def"] = ket["ke_mape_def"]
    return agg


# ── 三個 baseline（逐幀、單通道空間重建）──────────────────────────────
def rbf_multiquadric(pos, vals, grid_xy, epsilon=10.0):
    # scipy.interpolate.RBFInterpolator multiquadric（穩定 solve）。舊 Rbf 全域 smooth
    # 核在 K=100 稀疏點上矩陣近奇異 → 數值爆震，故用 RBFInterpolator。
    from scipy.interpolate import RBFInterpolator
    return RBFInterpolator(pos, vals, kernel="multiquadric", epsilon=epsilon)(grid_xy)


def idw(pos, vals, grid_xy, p=2.0, eps=1e-12):
    d = np.linalg.norm(grid_xy[:, None, :] - pos[None, :, :], axis=-1)  # [M,K]
    exact = d < eps
    w = 1.0 / (d ** p + eps)
    out = (w @ vals) / w.sum(axis=1)
    hit = exact.any(axis=1)
    if hit.any():
        out[hit] = vals[exact[hit].argmax(axis=1)]
    return out


def _divfree_modes(kmax):
    modes = []
    for m in range(-kmax, kmax + 1):
        for n in range(-kmax, kmax + 1):
            if m == 0 and n == 0:
                continue
            if m > 0 or (m == 0 and n > 0):
                modes.append((m, n))
    return modes


#: div-free trig LSQ 的奇異值截斷門檻，相對 s_max。
#: 值由實測選定，不是猜的：本 repo 全部良態佈點（LES-QR / QR-pivot / K≥200 spacefill）
#: 的 s_min/s_max ≥ 8e-3，故 1e-3 不動它們一個模態；而晶格佈點被 1e-9…1e-5 擾動時
#: s_min/s_max 落在 1e-8…1.6e-4，1e-3 把整條近退化帶都截掉。門檻取得比該帶小
#: （例如 1e-6）會讓帶內的近零奇異值被反轉而非截斷，重建誤差回到 1e4 量級。
DEFAULT_TRIG_RCOND = 1e-3


def _trig_design(xy, modes):
    """[2K, ncol] 設計矩陣：上半列給 u、下半列給 v，最後兩行是均流 (u0, v0)。

    座標一律先轉 float64。載入端交出來的 sensor_pos 是 float32（data.py），
    在 float32 下算相位會給每個模態約 1e-7 的擾動——足以把「精確退化的晶格佈點」
    推進「近退化」區間，而近退化正是截斷門檻最難處理的地方（精確退化反而好處理，
    奇異值直接掉到 1e-16）。這個 dtype 是上游的實作細節，不該決定論文數字。
    """
    xy = np.asarray(xy, dtype=np.float64)
    K = xy.shape[0]
    xs, ys = xy[:, 0], xy[:, 1]
    ncol = 2 * len(modes) + 2
    Au = np.zeros((K, ncol)); Av = np.zeros((K, ncol))
    for j, (m, n) in enumerate(modes):
        th = 2 * np.pi * (m * xs + n * ys)
        s, c = np.sin(th), np.cos(th)
        Au[:, 2 * j] = -4 * np.pi * n * s
        Au[:, 2 * j + 1] = -4 * np.pi * n * c
        Av[:, 2 * j] = 4 * np.pi * m * s
        Av[:, 2 * j + 1] = 4 * np.pi * m * c
    Au[:, -2] = 1.0
    Av[:, -1] = 1.0
    return np.vstack([Au, Av])


def trig_design_diagnostics(pos_uv, kmax=5, rcond=DEFAULT_TRIG_RCOND) -> dict:
    """佈點對 div-free trig 基底的可解性：條件數與截斷後的有效秩。

    設計矩陣只由佈點決定、與時間無關，故整段評估算一次就夠。近退化在這裡就看得見，
    不必等 metric 冒出 1e4 才發現——那個量級在表格裡看起來只是「這個 baseline 很爛」。
    """
    A = _trig_design(pos_uv, _divfree_modes(kmax))
    s = np.linalg.svd(A, compute_uv=False)
    return {
        "trig_rcond": float(rcond),
        "trig_cond": float(s[0] / s[-1]) if s[-1] > 0 else float("inf"),
        "trig_rank": int((s > rcond * s[0]).sum()),
        "trig_ncol": int(A.shape[1]),
    }


def divfree_trig_lsq(pos_uv, u_s, v_s, grid_xy, kmax=5, rcond=DEFAULT_TRIG_RCOND):
    """stream function ψ=Σ ψ_k e^{2πik·x}，u=∂ψ/∂y，v=-∂ψ/∂x → 自動無散度。
    未知＝各模態 ψ_k 實/虛部 ＋ 均流 (u0,v0)。LSQ 擬合感測 (u,v)。

    解法是截斷 SVD（`lstsq` 的 rcond 是相對 s_max 的門檻），不是 `A^T A + 絕對 ridge`。
    絕對 ridge 不隨奇異譜縮放：佈點接近規則晶格時設計矩陣秩虧損（1/16 棋盤子晶格下
    kmax=5 帶內的六個對角模態線性相依，rank 116/122），那些近零奇異值恰好落在 ridge
    之上就被「反轉」而非截斷，係數暴增——實測 ||coef|| 5.7e4、重建 rel-L2 7.4e4。
    直接分解 A 而非 A^T A 也避免把條件數平方。
    """
    modes = _divfree_modes(kmax)
    A = _trig_design(pos_uv, modes)
    b = np.concatenate([u_s, v_s])
    coef = np.linalg.lstsq(A, b, rcond=rcond)[0]
    G = _trig_design(grid_xy, modes)
    pred = G @ coef
    M = grid_xy.shape[0]
    return pred[:M], pred[M:]


def _resolve_les_path(path):
    """供 provenance 記錄用：回傳與 load_les_basis 相同的解析結果。"""
    from pi_lnn_jax.data import _resolve_data_path
    return _resolve_data_path(path).resolve()


def load_les_basis(path, n_grid, time_stride):
    """載入 placement LES 全場並下採樣到 eval 格點，供 gappy-POD 建 basis。

    Why LES not DNS: 這張表的前提是 sensor-only、無 DNS 存取。用 DNS 建 basis 會讓
    gappy 讀到它要重建的那個場，論文他處已把那條路標為 upper reference 而非可部署方法。

    檔案是 pickled dict {time, u, v}，u/v 為 [M, N, N]。格點不整除即 fail-fast——
    靜默重採樣會讓 basis 與 eval 落在不同格上，而形狀恰好相容時不會 crash。
    """
    from pi_lnn_jax.data import _resolve_data_path
    # 走 repo 的解析順序（PILNJAX_ROOT → PILNJAX_DATA_ROOT），否則相對路徑在
    # 大檔池與 checkout 分離的機器上開不起來——job 5410 就是這樣 fail 的。
    resolved = _resolve_data_path(path)
    raw = np.load(resolved, allow_pickle=True)
    d = raw.item() if raw.dtype == object else raw
    u = np.asarray(d["u"], dtype=np.float64)
    v = np.asarray(d["v"], dtype=np.float64)
    if u.ndim != 3 or u.shape != v.shape:
        raise ValueError(f"LES u/v 需同形 [M,N,N]，得 {u.shape} / {v.shape}")
    n_les = u.shape[1]
    if n_les % n_grid:
        raise ValueError(f"LES 格點 {n_les} 無法整除到 eval 格點 {n_grid}")
    s = n_les // n_grid
    return u[::time_stride, ::s, ::s], v[::time_stride, ::s, ::s]


def reference_time_mean(dns_u, dns_v):
    """參考場在評估窗上的逐格時間平均，排法與 `GappyPOD.fit` 的 mean 相同 [2N^2, 1]。

    供投影地板的對照用：把 LES 的時均場換成 DNS 的，其餘（模態）不變，
    藉此分離「基底方向不對」與「兩個流場的平均量不同」。
    """
    dns_u = np.asarray(dns_u, dtype=np.float64)
    dns_v = np.asarray(dns_v, dtype=np.float64)
    T = dns_u.shape[0]
    stacked = np.stack([dns_u.reshape(T, -1), dns_v.reshape(T, -1)], axis=1)  # [T,2,N^2]
    return stacked.reshape(T, -1).mean(axis=0)[:, None]                       # [2N^2,1]


def project_reference_onto_basis(modes, mean, dns_u, dns_v):
    """把參考場正交投影到 POD 基底，回傳 (u, v) [T,N,N]——係數擬合完美的上界。

    `modes` 的欄取自 SVD 的 U，彼此正交且單位長，故正交投影就是
    `mean + Φ Φᵀ (x − mean)`，不必解最小平方（解了也是同一個答案，
    只是多一份數值誤差來源）。向量排法與 `GappyPOD.fit` 相同：
    `[u_flat; v_flat]`（row-major）——兩者若不同排法，地板與 gappy 那一列
    就不可比，而那正是引用它的唯一理由。
    """
    modes = np.asarray(modes, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    dns_u = np.asarray(dns_u, dtype=np.float64)
    dns_v = np.asarray(dns_v, dtype=np.float64)
    T, N = dns_u.shape[0], dns_u.shape[1]
    if mean.shape != (2 * N * N, 1) or modes.shape[0] != 2 * N * N:
        raise ValueError(
            f"基底與參考場格點不符：modes {modes.shape} / mean {mean.shape} vs N={N}")
    X = np.stack([dns_u.reshape(T, -1), dns_v.reshape(T, -1)], axis=1).reshape(T, -1).T  # [2N^2,T]
    Xp = mean + modes @ (modes.T @ (X - mean))
    return (Xp[: N * N].T.reshape(T, N, N).copy(),
            Xp[N * N:].T.reshape(T, N, N).copy())


def _selftest():
    rng = np.random.default_rng(0)
    N = 32
    xs = np.linspace(0, 1, N, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u_t = -2 * np.pi * 2 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * 2 * Y)
    v_t = -2 * np.pi * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * 2 * Y)
    idx = rng.choice(N * N, 100, replace=False)
    pos = np.stack([X.ravel()[idx], Y.ravel()[idx]], 1)
    grid = np.stack([X.ravel(), Y.ravel()], 1)
    ur, vr = divfree_trig_lsq(pos, u_t.ravel()[idx], v_t.ravel()[idx], grid, kmax=5)
    err = np.sqrt(((ur - u_t.ravel()) ** 2).mean()) / np.sqrt((u_t ** 2).mean())
    ug = ur.reshape(N, N); vg = vr.reshape(N, N)
    k = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    div_hat = 1j * KX * np.fft.fft2(ug) + 1j * KY * np.fft.fft2(vg)
    vscale = np.sqrt((u_t ** 2 + v_t ** 2).mean())
    div = np.abs(np.fft.ifft2(div_hat)).mean() / vscale
    print(f"[selftest] divfree-trig recon u rel-L2={err:.2e}  spectral |div|/|v|={div:.2e}")
    assert err < 1e-3, "div-free trig 未能重建已知無散度場"
    assert div < 1e-9, "重建場非無散度（構造 bug）"
    print("[selftest] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-json")
    ap.add_argument("--dns-path")
    ap.add_argument("--config", default=None,
                    help="訓練 config TOML：follow_training 由它的 time_strides 決定取樣。"
                         "sensor/dns 仍由 --sensor-json/--dns-path 明示（本腳本比的是"
                         "單一 Re 的 classical 重建）")
    ap.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode],
                    help="評估協定（必填，無預設）")
    ap.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    ap.add_argument("--sensor-time-stride", type=int, default=None,
                    help="明示 stride；follow_training 下與 config 不一致即失敗")
    ap.add_argument("--sensor-T", type=int, default=200)
    ap.add_argument("--grid-stride", type=int, default=1)
    ap.add_argument("--max-grid", type=int, default=0)
    ap.add_argument("--rbf-eps", type=float, default=10.0)
    ap.add_argument("--idw-p", type=float, default=2.0)
    ap.add_argument("--trig-kmax", type=int, default=5)
    ap.add_argument("--trig-rcond", type=float, default=DEFAULT_TRIG_RCOND,
                    help="div-free trig LSQ 的奇異值截斷門檻（相對 s_max）。"
                         "預設同時涵蓋良態與近退化佈點；調小會讓近退化佈點的近零"
                         "奇異值被反轉而非截斷（係數暴增），調大會截掉良態佈點的真實模態")
    ap.add_argument("--les-basis", default=None,
                    help="placement LES 全場 .npy（pickled {time,u,v}）。給了才跑 gappy-POD；"
                         "不給則本腳本行為與新增此旗標前逐字相同。basis 只能是 LES——"
                         "用 DNS 建 basis 會讓這張表失去 sensor-only 前提")
    ap.add_argument("--les-time-stride", type=int, default=5,
                    help="LES 快照時間下採樣（控 SVD 記憶體）；basis 是統計子空間，"
                         "不與 eval 幀對齊，故下採樣不影響公平性")
    ap.add_argument("--pod-modes", type=int, default=0,
                    help="0＝以 basis 自身切 fit/val 選模態（leakage-free）；>0 則固定該模態數")
    ap.add_argument("--projection-floor", action="store_true",
                    help="另外評估參考場對 --les-basis 那組 POD 基底的正交投影，"
                         "即係數擬合完美時仍會留下的誤差。**它讀參考場**，不是 "
                         "sensor-only baseline；產出以 les_projection_floor 命名並在 "
                         "provenance 標 reads_reference_field=true。需 --les-basis")
    ap.add_argument("--projection-floor-modes", type=int, nargs="+", default=None,
                    help="地板要評的模態數（可多個，一次 SVD 掃完）。未給則沿用 gappy "
                         "那一列的秩。與 --pod-modes 分開是刻意的：地板要掃到 K 撐不起的"
                         "秩（400、800）才看得出基底的極限，而同一個秩會讓 gappy 那一列"
                         "欠定，兩者不該被同一個旋鈕綁住")
    ap.add_argument("--projection-floor-dns-mean", action="store_true",
                    help="地板另評一組把 LES 時均場換成 DNS 時均場的變體（模態不變）。"
                         "用來分離「基底方向不對」與「兩個流場的平均量不同」")
    ap.add_argument("--re", type=float, default=None,
                    help="Reynolds number；ν=1/Re 供 band/γ 的 k_η 邊界使用。未給則不產出 band/γ 欄位（不猜、不套預設）")
    ap.add_argument("--output", default="artifacts/baseline_eval/tab_fair_metrics.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return

    from pi_lnn_jax.evaluation_protocol import (
        load_for_evaluation,
        resolve_protocol,
        training_time_strides_from_config,
    )
    from pi_lnn_jax.evaluation_run import EvaluationRunRecorder, RunArtifactIdentity
    from pi_lnn_jax.metric_artifact import write_compatibility_projection

    nu_for_ctx = None if a.re is None else 1.0 / float(a.re)
    protocol = resolve_protocol(
        mode=a.protocol,
        training_time_strides=training_time_strides_from_config(a.config) if a.config else [],
        cli_time_stride=a.sensor_time_stride, sensor_T=a.sensor_T,
        reason=a.protocol_reason)
    d = load_for_evaluation(a.sensor_json, a.dns_path, protocol=protocol,
                            viscosity=nu_for_ctx,
                            grid_stride=a.grid_stride, max_grid=a.max_grid)
    sensor_phys, pos = d.sensor_phys, d.sensor_pos
    dns_u, dns_v = d.dns_u_eval, d.dns_v_eval
    T, Np = d.T, d.Nprime
    xs = np.linspace(0.0, 1.0, Np, endpoint=False).astype(np.float64)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([XX.ravel(), YY.ravel()], 1)
    print(f"[data] T={T} K={pos.shape[0]} grid={Np}x{Np}  "
          f"protocol={protocol.mode.value} stride={protocol.sensor_time_stride}")

    methods = {
        "rbf_multiquadric": lambda p, u, v: (
            rbf_multiquadric(p, u, grid, a.rbf_eps), rbf_multiquadric(p, v, grid, a.rbf_eps)),
        "idw": lambda p, u, v: (idw(p, u, grid, a.idw_p), idw(p, v, grid, a.idw_p)),
        "divfree_trig_lsq": lambda p, u, v: divfree_trig_lsq(
            p, u, v, grid, a.trig_kmax, a.trig_rcond),
    }
    # 佈點的可解性只由佈點決定，逐幀不變 → 算一次、印出來、進 provenance。
    trig_diag = trig_design_diagnostics(pos, kmax=a.trig_kmax, rcond=a.trig_rcond)
    print(f"[trig] cond={trig_diag['trig_cond']:.3e} "
          f"rank={trig_diag['trig_rank']}/{trig_diag['trig_ncol']} "
          f"rcond={trig_diag['trig_rcond']:.1e}")
    if trig_diag["trig_rank"] < trig_diag["trig_ncol"]:
        print(f"[warn] div-free trig 設計矩陣秩虧損 "
              f"({trig_diag['trig_ncol'] - trig_diag['trig_rank']} 個方向被截斷)："
              f"佈點無法分辨 kmax={a.trig_kmax} 帶內的全部模態（規則晶格的典型徵狀）。"
              f"該 baseline 的誤差含此不可識別性，不只是方法本身的能力")
    # gappy-POD 是 opt-in：不給 --les-basis 時 methods 與此旗標存在前逐字相同，
    # 既有三列因此可逐位元回歸。
    gappy_details: dict = {}
    if a.les_basis:
        from pi_lnn_jax.baselines import GappyPOD
        les_u, les_v = load_les_basis(a.les_basis, Np, a.les_time_stride)
        if a.pod_modes > 0:
            r, mode_src = int(a.pod_modes), "cli"
        else:
            from pi_lnn_jax.baseline_eval import select_pod_modes_by_validation
            # 掃描格與 evaluate_baselines.py / eval_gappy_cross_re.py 逐字相同：
            # 三支腳本挑模態的方式必須一致，否則 gappy 的「秩」在不同表裡意義不同。
            _K = int(pos.shape[0])
            _grid = [max(1, _K // 8), _K // 4, _K // 2, _K, 3 * _K // 2, 2 * _K]
            r, _curve = select_pod_modes_by_validation(
                les_u, les_v, pos, _grid, val_frac=0.3, seed=0)
            mode_src = "validation"
        _gp = GappyPOD(n_modes=r).fit(les_u, les_v)
        print(f"[gappy] basis={les_u.shape} modes={r} ({mode_src}) from {a.les_basis}")

        def _gappy(p, u_s, v_s, _m=_gp):
            ur, vr = _m.reconstruct(np.stack([u_s, v_s], 1)[None], p)
            return ur[0].astype(np.float64).ravel(), vr[0].astype(np.float64).ravel()

        methods["gappy_pod_les"] = _gappy
        gappy_details = {
            "les_basis": str(_resolve_les_path(a.les_basis)),
            "les_time_stride": a.les_time_stride,
            "les_basis_snapshots": int(les_u.shape[0]),
            "pod_modes": r,
            "pod_modes_source": mode_src,
        }
    # ν=1/Re（physics.py 慣例）。未給 --re 時 band/γ 欄位缺席——不從 dns_path 檔名
    # 猜 Re，那是 eval 腳本的靜默錯配來源。
    nu = None if a.re is None else 1.0 / float(a.re)
    if nu is None:
        print("[warn] 未給 --re → 不產出 band_rel_err_*/gamma_* 欄位")

    context = d.context
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 一個 evaluation run 建一次 recorder：它自建構時抓一次 code revision（與遷移前
    # 的 repository_revision(_REPO_ROOT) 同一個 repo root），並擁有 stamp/命名/寫盤。
    # out_path 刻意不 .resolve()——傳未解析的它，讓寫出的檔名與舊版逐字相同。
    recorder = EvaluationRunRecorder("scripts/classical_baselines_fair.py", out_path)

    out = {}
    for name, fn in methods.items():
        u_rec = np.empty((T, Np, Np)); v_rec = np.empty((T, Np, Np))
        for t in range(T):
            u_s, v_s = sensor_phys[t, :, 0], sensor_phys[t, :, 1]
            ur, vr = fn(pos, u_s, v_s)
            u_rec[t] = ur.reshape(Np, Np); v_rec[t] = vr.reshape(Np, Np)

        # method 與重建超參是這筆記錄的來源身分：同一組 sensor/DNS/時間軸下，
        # 唯一區分三份 artifact 的就是它們。recorder 自行蓋上 evaluation_protocol
        # （caller 不得再供），並依 identity 重現凍結檔名 {stem}_{method}（單-Re、無 _re）。
        projection = recorder.record(
            u_rec, v_rec, dns_u, dns_v,
            context=context, protocol=protocol,
            identity=RunArtifactIdentity(method=name),
            inputs=(
                ("sensor", str(Path(a.sensor_json).resolve())),
                ("dns", str(Path(a.dns_path).resolve())),
            ),
            details_extras={
                "method": name,
                "reynolds": a.re,
                "sensor_time_stride": a.sensor_time_stride,
                "sensor_T": a.sensor_T,
                "grid_stride": a.grid_stride,
                "max_grid": a.max_grid,
                "K": int(pos.shape[0]),
                "rbf_eps": a.rbf_eps,
                "idw_p": a.idw_p,
                "trig_kmax": a.trig_kmax,
                **(trig_diag if name == "divfree_trig_lsq" else {}),
                **(gappy_details if name == "gappy_pod_les" else {}),
            },
        )
        print(f"[out] canonical artifact written ({name})")

        agg = method_agg(projection, banded=nu is not None)
        out[name] = agg
        print(f"[{name}] KE-MAPE={agg['ke_t_mape']*100:.2f}%  uv={agg['uv_rel_err']*100:.2f}%  "
              f"u={agg['u_rel_err']*100:.2f}%  v={agg['v_rel_err']*100:.2f}%  "
              f"w={agg['omega_rel_err']*100:.2f}%")
    # LES 子空間的投影地板：把參考場正交投影到 gappy-POD 所用的同一組基底，
    # 得到「係數擬合完美時仍會留下的誤差」。基底刻意共用 _gp 而非重建一份——
    # 兩者若不同源，地板與 gappy 那一列就不可比，而那正是引用它的唯一理由。
    # 它讀參考場，因此不是 sensor-only baseline；命名與 provenance 都標明。
    if a.projection_floor:
        if not a.les_basis:
            raise SystemExit("--projection-floor 需要 --les-basis（地板由該基底定義）")
        floor_ranks = (sorted({int(r) for r in a.projection_floor_modes})
                       if a.projection_floor_modes else [int(_gp.modes.shape[1])])
        if floor_ranks[0] < 1:
            raise SystemExit(f"--projection-floor-modes 須為正整數，收到 {a.projection_floor_modes}")
        need = floor_ranks[-1]
        # 更深的截斷仍是同一份 SVD 的前 need 欄，故 basis 與 gappy 那一列同源；
        # `_gp` 夠深時直接切它，避免白算第二次 SVD。
        gp_floor = _gp if _gp.modes.shape[1] >= need else GappyPOD(n_modes=need).fit(les_u, les_v)
        avail = int(gp_floor.modes.shape[1])
        if avail < need:
            # GappyPOD._select_r 對超額的請求是靜默 min()——那會讓「800 模態的地板」
            # 其實是 500 模態的地板，而數字看起來完全正常。這裡把它變成失敗。
            raise SystemExit(
                f"--projection-floor-modes 要 {need} 個模態，但 {les_u.shape[0]} 個 LES 快照"
                f"的 basis 只有 {avail} 欄。降低模態數，或用更小的 --les-time-stride")
        variants = [("les", gp_floor.mean)]
        if a.projection_floor_dns_mean:
            variants.append(("dns", reference_time_mean(dns_u, dns_v)))
        for r in floor_ranks:
            for mean_src, mean_vec in variants:
                name = ("les_projection_floor" if mean_src == "les"
                        else "les_projection_floor_dnsmean")
                u_rec, v_rec = project_reference_onto_basis(
                    gp_floor.modes[:, :r], mean_vec, dns_u, dns_v)
                projection = recorder.record(
                    u_rec, v_rec, dns_u, dns_v,
                    context=context, protocol=protocol,
                    identity=RunArtifactIdentity(method=name, modes=r),
                    inputs=(
                        ("sensor", str(Path(a.sensor_json).resolve())),
                        ("dns", str(Path(a.dns_path).resolve())),
                    ),
                    details_extras={
                        "method": name,
                        "reynolds": a.re,
                        "sensor_time_stride": a.sensor_time_stride,
                        "sensor_T": a.sensor_T,
                        "grid_stride": a.grid_stride,
                        "max_grid": a.max_grid,
                        "K": int(pos.shape[0]),
                        "reads_reference_field": True,
                        **gappy_details,
                        # 地板的秩與均值來源是它自己的身分，覆蓋 gappy 那一列的值。
                        "pod_modes": r,
                        "pod_modes_source": ("projection_floor_modes"
                                             if a.projection_floor_modes else mode_src),
                        "projection_mean_source": mean_src,
                    },
                )
                agg = method_agg(projection, banded=nu is not None)
                out[f"{name}_m{r}"] = agg
                print(f"[{name}_m{r}] uv={agg['uv_rel_err']*100:.2f}%  "
                      f"u={agg['u_rel_err']*100:.2f}%  v={agg['v_rel_err']*100:.2f}%  "
                      f"KE-MAPE={agg['ke_t_mape']*100:.2f}%")
    write_compatibility_projection(out_path, out)
    print(f"[out] {a.output}")


if __name__ == "__main__":
    main()

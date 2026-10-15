"""diag_unseen_window.py — 已訓練的模型能否不重訓、直接重建「沒見過的」感測串流。

What:
    審稿人（R1/R2/R3）：每個案例都在同一條軌跡上擬合與評估，「operator」與低延遲的定位
    沒有證據。本診斷把訓練好的模型（不改任何參數）餵另一段時間窗的感測值：
      * 控制窗 [0, 5]：取自 T=10 DNS 的前段——就是訓練用的那段軌跡。必須重現該 run
        既有評估的 uv（證明感測點抽取、時間對齊、正規化都與訓練一致）。
      * 未見窗 [5, 10]：同一流場、同一感測點、模型從沒看過的後段。
    兩窗都把感測時間標成 [0, 5]（模型的時間座標是窗內相對時間；強迫與 Re 不隨時間變）；
    正規化一律用**訓練時的** norm_stats（部署時沒有別的可用）。
    參照：同一批未見窗感測值上的圓盤 div-free LSQ（kc 以 leave-sensors-out CV 選定，
    與 classical_baselines_fair --extended 同一套），回答「不重訓的 PI-CON 是否仍勝過純空間插值」。

Why T=10 DNS:
    `..._T10_..._cuda.npy` 是 T=5 那條軌跡的延長（verify.json：[0,5] 逐幀吻合，
    knowledge/experiments/kolmogorov-temporal-extrapolation-2026-08.md「前提」節）。
    本腳本仍在執行時重驗：T=10 檔前 101 幀與評估用 T=5 DNS 的相對差須 < DNS_MATCH_TOL。

這是**診斷**不是 eval producer：不落 metric artifact，輸出只寫到 `--out`。

Usage（與 eval_pressure_limit 相同的參數，外加 --dns-t10）:
    PYTHONPATH=. uv run python scripts/diag_unseen_window.py --dns-t10 <T10.npy> \
        --arch liquid --protocol follow_training --artifacts-dir <run> --config <toml> \
        --expect-metrics <run>/final_eval/metrics.json --arm K100_al_on --out <json>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classical_baselines_fair as cbf
import diag_pressure_poisson_jump as dpj
import eval_pressure_limit as epl

from pi_lnn_jax.data import load_dns_from_path
from pi_lnn_jax.metric_artifact import compute_metrics

PRODUCER = "scripts/diag_unseen_window.py"
SCHEMA = "unseen_window.v1"
#: T=10 檔前段與 T=5 評估 DNS 的逐幀相對 L2 上限。兩檔由不同硬體生成（_cuda），
#: verify.json 記錄的吻合程度遠低於此；超過代表不是同一條軌跡，整個比較失效。
DNS_MATCH_TOL = 1e-3
#: 感測點在 T=10 檔上抽出的值與訓練感測值的相對差上限（同一軌跡、同一格點時應為 ~0）。
SENSOR_MATCH_TOL = 1e-3
WINDOW_LEN = 5.0
LSQ_KC = (3, 4, 5, 6, 7, 8)


def grid_indices(sensor_pos: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """感測座標（[0,1) 週期）→ 最近的格點索引 (ix, iy)；座標須落在格點上（容差 1e-6）。"""
    pos = np.asarray(sensor_pos, dtype=np.float64)
    f = pos * N
    idx = np.rint(f).astype(int) % N
    off = np.abs(f - np.rint(f)).max()
    if off > 1e-4:
        raise ValueError(f"感測點不在 {N}² 格點上（最大偏移 {off:.2e} 格）——抽值規則不成立")
    return idx[:, 0], idx[:, 1]


def window_frames(t: np.ndarray, start: float, n: int) -> np.ndarray:
    """從 start 起連續 n 幀的索引；start 必須恰為某一幀（容差 1e-6）。"""
    i0 = int(np.argmin(np.abs(t - start)))
    if abs(t[i0] - start) > 1e-6:
        raise ValueError(f"t={start} 不是 DNS 的幀時刻")
    if i0 + n > len(t):
        raise ValueError(f"窗 [{start}, +{n} 幀] 超出 DNS 長度 {len(t)}")
    return np.arange(i0, i0 + n)


def rel_l2(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def run(args) -> dict:
    t_start = time.time()
    ctx = dpj.prepare(args)
    al = ctx.aligned
    n = len(ctx.times)
    st = np.asarray(al.sensor_time, dtype=np.float64)
    if len(st) != n or not np.allclose(st, ctx.times, atol=1e-6) or abs(st[-1] - WINDOW_LEN) > 1e-6:
        raise ValueError("本診斷假設感測時刻 = 評估時刻 = [0, 5] 上的 101 幀")
    dt = float(np.diff(st).mean())

    u10, v10, t10 = load_dns_from_path(str(args.dns_t10), time_stride=1)
    u10 = np.asarray(u10, np.float64); v10 = np.asarray(v10, np.float64)
    t10 = np.asarray(t10, np.float64)
    stride = round(dt / float(np.diff(t10).mean()))
    if stride < 1 or abs(stride * float(np.diff(t10).mean()) - dt) > 1e-6:
        raise ValueError(f"T=10 DNS 幀距 {np.diff(t10).mean()} 無法整除感測間隔 {dt}")
    u10, v10, t10 = u10[::stride], v10[::stride], t10[::stride]
    if u10.shape[1:] != al.dns_u_eval.shape[1:]:
        raise ValueError(f"T=10 DNS 格點 {u10.shape[1:]} 與評估 DNS {al.dns_u_eval.shape[1:]} 不同")

    ctrl = window_frames(t10, 0.0, n)
    unseen = window_frames(t10, WINDOW_LEN, n)
    dns_match = max(rel_l2(np.stack([u10[i], v10[i]]), np.stack([al.dns_u_eval[k], al.dns_v_eval[k]]))
                    for k, i in enumerate(ctrl))
    if dns_match > DNS_MATCH_TOL:
        raise SystemExit(f"[FAIL] T=10 DNS 前段與評估 DNS 不吻合（max rel {dns_match:.2e}）——不是同一條軌跡")

    ix, iy = grid_indices(np.asarray(al.sensor_pos), u10.shape[1])

    def sensors(frames):
        return np.stack([u10[frames][:, ix, iy], v10[frames][:, ix, iy]], axis=-1)   # [T,K,2]

    s_ctrl = sensors(ctrl)
    sensor_match = rel_l2(s_ctrl, np.asarray(al.sensor_phys, np.float64))
    if sensor_match > SENSOR_MATCH_TOL:
        raise SystemExit(f"[FAIL] T=10 抽出的感測值與訓練感測值相對差 {sensor_match:.2e}——抽值規則不符")

    ns = ctx.norm_stats
    mean = np.array([ns["u_mean"], ns["v_mean"]]); std = np.array([ns["u_std"], ns["v_std"]])

    def reconstruct(s_phys, frames):
        sv = jnp.asarray(((s_phys - mean) / std).astype(np.float32))
        errs = []
        it = epl.iter_uvp_frames(ctx.model, ctx.params, sv, ctx.sensor_pos, ctx.re_norm,
                                 ctx.sensor_time, ns, ctx.N, ctx.times)
        for k, (u, v, _p) in enumerate(it):
            errs.append(compute_metrics(u, v, u10[frames[k]], v10[frames[k]])["uv_rel_err"])
        return np.asarray(errs)

    e_ctrl = reconstruct(s_ctrl, ctrl)
    gate_diff = abs(float(e_ctrl.mean()) - ctx.expect.uv_rel_err)
    gate = {"expected": ctx.expect.uv_rel_err, "control_window": float(e_ctrl.mean()),
            "abs_diff": gate_diff, "atol": epl.UV_GATE_ATOL, "passed": gate_diff <= epl.UV_GATE_ATOL}
    print(f"[gate] 控制窗 uv {e_ctrl.mean():.6f} vs 既有 {ctx.expect.uv_rel_err:.6f} |Δ|={gate_diff:.2e}")
    if not gate["passed"]:
        raise SystemExit("[FAIL] 控制窗重現不了既有評估——感測、對齊或正規化與訓練不一致，未寫輸出")

    s_unseen = sensors(unseen)
    e_unseen = reconstruct(s_unseen, unseen)
    print(f"[unseen] uv mean {e_unseen.mean():.4f}  (控制窗 {e_ctrl.mean():.4f})")

    # 參照：圓盤 div-free LSQ，kc 由未見窗感測值的 leave-sensors-out CV 選（不讀 DNS 場）
    pos = np.asarray(al.sensor_pos, np.float64)
    N = ctx.N
    xs = np.arange(N) / N
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([XX.ravel(), YY.ravel()], 1)

    def make(kc):
        m = cbf._divfree_modes_disk(kc)
        return lambda ptr, utr, vtr, pte: cbf.divfree_trig_lsq(ptr, utr, vtr, pte, modes=m)

    kc, curve = cbf.select_by_sensor_cv(make, list(LSQ_KC), pos, s_unseen,
                                        list(range(0, n, 10)), folds=10)
    modes = cbf._divfree_modes_disk(kc)
    e_lsq = []
    for k, i in enumerate(unseen):
        ur, vr = cbf.divfree_trig_lsq(pos, s_unseen[k, :, 0], s_unseen[k, :, 1], grid, modes=modes)
        e_lsq.append(compute_metrics(ur.reshape(N, N), vr.reshape(N, N), u10[i], v10[i])["uv_rel_err"])
    e_lsq = np.asarray(e_lsq)
    print(f"[lsq] kc={kc} uv mean {e_lsq.mean():.4f}")

    return {"producer": PRODUCER, "schema": SCHEMA, **dpj.provenance(ctx, args, Path(__file__)),
            "dns_t10": str(args.dns_t10), "dns_match_max_rel": dns_match,
            "sensor_match_rel": sensor_match, "uv_gate": gate,
            "control_window": {"t_range": [float(t10[ctrl[0]]), float(t10[ctrl[-1]])],
                               "uv_mean": float(e_ctrl.mean()), "per_frame": e_ctrl.tolist()},
            "unseen_window": {"t_range": [float(t10[unseen[0]]), float(t10[unseen[-1]])],
                              "uv_mean": float(e_unseen.mean()), "per_frame": e_unseen.tolist()},
            "lsq_unseen": {"kc": kc, "cv_curve": {str(c): e for c, e in curve.items()},
                           "uv_mean": float(e_lsq.mean()), "per_frame": e_lsq.tolist()},
            "wall_seconds": time.time() - t_start}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--dns-t10" not in argv:
        raise SystemExit("--dns-t10 必填")
    i = argv.index("--dns-t10")
    dns_t10 = Path(argv[i + 1]); del argv[i:i + 2]
    args = epl.parse_args(argv)
    args.dns_t10 = dns_t10
    out = run(args)
    text = json.dumps(out, indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

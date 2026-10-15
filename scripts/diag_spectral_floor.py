"""diag_spectral_floor.py — 重建能譜高波數平台（PoF Fig. 5 / Fig. 9）的來源判別。

What:
    PI-CON 的徑向能譜在 k≳25 平在約 1e-10、到網格角落 k≈181 才驟降，三個 K 同形
    （審稿人 R1/R2/R3/R5 共同提出）。本診斷以同一顆 ckpt、同一組時刻重建 (u, v)，
    改變兩個與物理無關的設定，看平台跟著哪一個動：

      precision  fp32（訓練與既有評估的精度）vs fp64（參數、輸入、查詢點全部升精度）
      grid       256²（DNS 網格，既有評估）vs 512²（兩倍解析度的查詢點）

    判讀（事前登錄，結果出來前寫下）：
      * fp64 的平台比 fp32 低數個數量級 → 平台是 float32 數值底噪，不是模型內容。
      * 512² 在 k∈[30, 90] 的能量明顯低於 256²，且 k>128 出現能量 → 平台含 256² 取樣
        折疊回來的超 Nyquist 內容（aliasing）。
      * 兩者都不動 → 平台是模型在連續座標上真實表達的高頻內容（例：trunk MLP 非線性
        合成的諧波），需由架構面解釋。

Why 兩次執行而不是一次:
    jax 的 x64 必須在任何陣列建立前決定（JAX_ENABLE_X64 環境變數），同一個程序內不能
    切換。sbatch 以兩次呼叫分別跑 fp32 與 fp64；本腳本斷言環境與 --precision 一致。

閘門:
    fp32、256² 的每個時刻都與該 run 既有評估的逐幀 uv_rel_err 比對（容許 2e-4，
    同 eval_pressure_limit.UV_GATE_ATOL）；對不上就不寫輸出——重建的不是那個模型。
    fp64 不設閘門（它與 fp32 的差本身就是要量的東西），差值照實記錄。

這是**診斷**不是 eval producer：不落 metric artifact，輸出只寫到 `--out`。

Usage（與 eval_pressure_limit 相同的參數，外加 --precision 與 --times）:
    JAX_ENABLE_X64=1 PYTHONPATH=. uv run python scripts/diag_spectral_floor.py \
        --precision fp64 --times 2.5,5.0 --arch liquid --protocol follow_training \
        --artifacts-dir <run> --config <toml> --expect-metrics <run>/final_eval/metrics.json \
        --arm K200_al --out <job>/out/<run>_fp64.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_pressure_poisson_jump as dpj
import eval_pressure_limit as epl

from pi_lnn_jax.metric_artifact import compute_energy_spectrum, compute_metrics

PRODUCER = "scripts/diag_spectral_floor.py"
SCHEMA = "spectral_floor.v1"
GRIDS = (256, 512)
#: 平台的代表帶：Fig. 5 上平台從 k≈25 起、在 256² 的 Nyquist（128）以內。
PLATEAU_BAND = (30, 90)
#: 512² 才看得到的帶：256² 的 Nyquist 之上。
SUPER_NYQUIST_BAND = (140, 250)
DTYPES = {"fp32": (np.float32, jnp.float32), "fp64": (np.float64, jnp.float64)}


def band_median(E_k: np.ndarray, band: tuple[int, int]) -> float:
    """E(k) 在整數 shell [lo, hi] 的中位數；帶超出陣列長度時 fail-fast。"""
    lo, hi = band
    if hi >= len(E_k):
        raise ValueError(f"帶 {band} 超出 E(k) 長度 {len(E_k)}")
    return float(np.median(E_k[lo:hi + 1]))


def parse_times(spec: str, eval_times: np.ndarray) -> list[float]:
    """--times 的每個值都必須是評估時刻之一（容差 1e-6），否則沒有既有逐幀 uv 可對。"""
    out = []
    for s in spec.split(","):
        t = float(s)
        hit = np.abs(eval_times - t) <= 1e-6
        if not hit.any():
            raise ValueError(f"t={t} 不在評估時刻內")
        out.append(float(eval_times[np.argmax(hit)]))
    return out


def expected_uv_per_t(expect_path: Path, times: list[float]) -> dict[float, float]:
    """既有評估 metrics.json 的逐幀 uv_rel_err（provenance 已由 epl.read_expectation 核過）。"""
    rows = json.loads(Path(expect_path).read_text())["metrics_per_t"]
    out = {}
    for t in times:
        match = [r for r in rows if abs(float(r["t"]) - t) <= 1e-6]
        if len(match) != 1 or "uv_rel_err" not in match[0]:
            raise KeyError(f"{expect_path} 在 t={t} 找不到唯一且含 uv_rel_err 的列")
        out[t] = float(match[0]["uv_rel_err"])
    return out


def decode_uv(ctx, params, h_states, N: int, t: float, np_dtype, jnp_dtype):
    """在 N×N 查詢格點、時刻 t 重建 (u, v)；dtype 決定格點、時間與（呼叫端已轉的）參數精度。"""
    grid = np.linspace(0.0, 1.0, N, endpoint=False).astype(np_dtype)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    xy_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np_dtype)
    n_total = xy_np.shape[0]
    chunk = epl._query_chunk(int(ctx.aligned.K), n_total)
    xy = jnp.asarray(np.pad(xy_np, ((0, (-n_total) % chunk), (0, 0))), dtype=jnp_dtype)
    t_q = jnp.full((chunk,), t, dtype=jnp_dtype)
    sensor_time = ctx.sensor_time.astype(jnp_dtype)
    sensor_pos = ctx.sensor_pos.astype(jnp_dtype)
    outs = [epl._decode(ctx.model, params, xy[i:i + chunk], t_q, h_states, sensor_time, sensor_pos)
            for i in range(0, xy.shape[0], chunk)]
    if outs[0].dtype != jnp_dtype:
        raise TypeError(f"decode 輸出 dtype={outs[0].dtype}，預期 {jnp_dtype}——精度沒有傳到前向")
    pred = np.concatenate([np.asarray(o) for o in outs])[:n_total].astype(np.float64)
    ns = ctx.norm_stats
    u = pred[:, 0] * ns["u_std"] + ns["u_mean"]
    v = pred[:, 1] * ns["v_std"] + ns["v_mean"]
    return u.reshape(N, N), v.reshape(N, N)


def parse_args(argv=None):
    extra = [a for a in (argv if argv is not None else sys.argv[1:])]
    precision, times = None, None
    rest = []
    it = iter(extra)
    for a in it:
        if a == "--precision":
            precision = next(it)
        elif a == "--times":
            times = next(it)
        else:
            rest.append(a)
    if precision not in DTYPES:
        raise SystemExit(f"--precision 必填，須為 {sorted(DTYPES)}")
    if not times:
        raise SystemExit("--times 必填（逗號分隔的評估時刻）")
    args = epl.parse_args(rest)
    args.precision, args.times_spec = precision, times
    return args


def run(args) -> dict:
    t_start = time.time()
    want_x64 = args.precision == "fp64"
    if bool(jax.config.jax_enable_x64) != want_x64:
        raise SystemExit(f"--precision {args.precision} 但 JAX_ENABLE_X64={jax.config.jax_enable_x64}"
                         "——x64 必須以環境變數在程序啟動時設定")
    np_dtype, jnp_dtype = DTYPES[args.precision]
    ctx = dpj.prepare(args)
    times = parse_times(args.times_spec, ctx.times)
    expect_uv = expected_uv_per_t(ctx.expect.path, times)

    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp_dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, ctx.params)
    h_states = epl._encode(ctx.model, params, ctx.sensor_vals.astype(jnp_dtype),
                           ctx.sensor_pos.astype(jnp_dtype), ctx.re_norm,
                           ctx.sensor_time.astype(jnp_dtype))
    if h_states.dtype != jnp_dtype:
        raise TypeError(f"encode 輸出 dtype={h_states.dtype}，預期 {jnp_dtype}")

    t_index = {t: int(np.argmin(np.abs(ctx.times - t))) for t in times}
    results, gate_rows = [], []
    for t in times:
        i = t_index[t]
        u_ref, v_ref = ctx.aligned.dns_u_eval[i], ctx.aligned.dns_v_eval[i]
        _, E_dns = compute_energy_spectrum(u_ref, v_ref)
        for N in GRIDS:
            u, v = decode_uv(ctx, params, h_states, N, t, np_dtype, jnp_dtype)
            _, E_pred = compute_energy_spectrum(u, v)
            row = {"t": t, "grid": N, "E_pred_k": E_pred.tolist(),
                   "plateau_median": band_median(E_pred, PLATEAU_BAND)}
            if N == ctx.N:
                uv = compute_metrics(u, v, u_ref, v_ref)["uv_rel_err"]
                row["uv_rel_err"] = uv
                gate_rows.append({"t": t, "measured": uv, "expected": expect_uv[t],
                                  "abs_diff": abs(uv - expect_uv[t])})
            if N > ctx.N:
                row["super_nyquist_median"] = band_median(E_pred, SUPER_NYQUIST_BAND)
            results.append(row)
            print(f"  t={t:.3f} grid={N} {args.precision}  plateau[{PLATEAU_BAND}]="
                  f"{row['plateau_median']:.3e}"
                  + (f"  uv={row['uv_rel_err']:.6f}" if "uv_rel_err" in row else "")
                  + (f"  super-Nyquist={row['super_nyquist_median']:.3e}"
                     if "super_nyquist_median" in row else ""))
        results.append({"t": t, "grid": "dns", "E_dns_k": E_dns.tolist(),
                        "plateau_median": band_median(E_dns, PLATEAU_BAND)})

    gate = {"atol": epl.UV_GATE_ATOL, "applied": not want_x64, "rows": gate_rows,
            "passed": all(r["abs_diff"] <= epl.UV_GATE_ATOL for r in gate_rows)}
    if not want_x64 and not gate["passed"]:
        raise SystemExit(f"[FAIL] fp32 逐幀 uv 指紋不符：{gate_rows}——未寫輸出")
    return {"producer": PRODUCER, "schema": SCHEMA, **dpj.provenance(ctx, args, Path(__file__)),
            "precision": args.precision, "grids": list(GRIDS), "plateau_band": PLATEAU_BAND,
            "super_nyquist_band": SUPER_NYQUIST_BAND, "uv_gate": gate, "spectra": results,
            "wall_seconds": time.time() - t_start}


def main(argv=None) -> int:
    args = parse_args(argv)
    out = run(args)
    text = json.dumps(out, indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

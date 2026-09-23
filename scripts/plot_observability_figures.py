#!/usr/bin/env python3
"""plot_observability_figures.py — 產生 sec:placement / sec:ceiling 的可觀測性圖。

輸出（TMLR 風格，serif/向量 PDF）:
  figures/results/lespod_conditioning.pdf      — Fig A（sec:placement）
  figures/results/pressure_observability.pdf   — Fig B+C 雙 panel（sec:ceiling）

資料一律重算自 LES（DNS-free），重用 diag_observability / diag_pressure_observability。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.journal_style import setup_style, STYLE_CYCLE, save_figure  # noqa: E402

from scripts.diag_observability import (  # noqa: E402
    coords_to_xy_indices,
    divfree_half_modes,
    divfree_observation_operator,
    pod_basis,
    pod_observation_operator,
)
from scripts.diag_pressure_observability import (  # noqa: E402
    compute_Jp,
    project_velocity_to_coeffs,
)
import jax.numpy as jnp  # noqa: E402
import json  # noqa: E402

from pi_lnn_jax.data import _resolve_data_path  # noqa: E402

_DEFAULT_LES = str(_resolve_data_path("data/les/kolmogorov_les_Re10000_N256_T50_standalone.npy"))


def _cond(C):
    s = np.linalg.svd(C, compute_uv=False)
    return s, float(s[0] / s[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-json", default="data/sensors/re10000/sensors_les_qr_K100_N256_t0-20_si128.json")
    ap.add_argument("--les", default=_DEFAULT_LES)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--n-snapshots", type=int, default=8)
    ap.add_argument("--outdir", default="paper/tmlr-format/figures/results")
    args = ap.parse_args()

    coords = np.asarray(json.load(open(args.sensor_json))["selected_coordinates"], dtype=np.float64)
    K = coords.shape[0]
    raw = np.load(args.les, allow_pickle=True).item()
    u_all = np.asarray(raw["u"], dtype=np.float64)
    v_all = np.asarray(raw["v"], dtype=np.float64)
    N = u_all.shape[-1]
    print(f"[data] K={K}  LES {u_all.shape}  N={N}")

    half = divfree_half_modes(args.kmax)
    ix, iy = coords_to_xy_indices(coords, N)

    # ── LES-POD: κ(CΦ_r) dense curve + cumulative energy ──
    r_max = min(2 * K, u_all.shape[0])
    Phi, svals = pod_basis(u_all, v_all, r_max)
    cum_energy = np.cumsum(svals ** 2) / (svals ** 2).sum()
    r_grid = np.arange(2, r_max + 1, 2)
    kappa_r = np.array([_cond(pod_observation_operator(Phi[:, :r], ix, iy, N))[1] for r in r_grid])
    print(f"[POD] κ(r=100)={kappa_r[r_grid==100][0]:.2f}  energy(100)={cum_energy[99]*100:.2f}%  "
          f"κ(r=200)={kappa_r[-1]:.1f}")

    # ── velocity-only vs pressure-augmented singular spectra ──
    C_u, _ = divfree_observation_operator(coords, args.kmax)
    s_u, kappa_u = _cond(C_u)
    idx_snap = np.linspace(0, u_all.shape[0] - 1, args.n_snapshots).astype(int)
    spectra_up, kappa_up = [], []
    for i in idx_snap:
        a_bar = jnp.asarray(project_velocity_to_coeffs(u_all[i], v_all[i], half, N))
        Jp = compute_Jp(a_bar, half, ix, iy, N)
        C_up = np.concatenate([C_u, Jp], axis=0)
        s_up, k_up = _cond(C_up)
        spectra_up.append(s_up)
        kappa_up.append(k_up)
    kappa_up = np.array(kappa_up)
    # 代表性 snapshot：取 κ 中位數那個
    rep = int(np.argsort(kappa_up)[len(kappa_up) // 2])
    s_up_rep = spectra_up[rep]
    print(f"[pressure] velocity-only κ={kappa_u:.2f} rank={len(s_u)}; "
          f"pressure-aug rank={len(s_up_rep)} κ∈[{kappa_up.min():.0f},{kappa_up.max():.0f}]")

    setup_style("neurips")  # serif Times, font 9, no top/right spines, vector PDF
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ══ Figure A: LES-POD conditioning ══
    figA, axL = plt.subplots(figsize=(3.4, 2.6))
    cL = STYLE_CYCLE[0][0]
    cR = STYLE_CYCLE[1][0]
    axL.semilogy(r_grid, kappa_r, color=cL, lw=1.6)
    axL.set_xlabel(r"POD modes $r$")
    axL.set_ylabel(r"Conditioning $\kappa(C\Phi_r)$ (–)", color=cL)
    axL.tick_params(axis="y", labelcolor=cL)
    axL.axvline(2 * K, color="0.5", ls=":", lw=0.9)
    axL.text(2 * K - 4, kappa_r.max() * 0.4, r"$2K$", color="0.4", ha="right", fontsize=8)
    axL.plot(100, kappa_r[r_grid == 100][0], marker="o", color=cL, ms=5)
    axL.annotate(r"$\kappa{=}4.1$", (100, kappa_r[r_grid == 100][0]),
                 textcoords="offset points", xytext=(6, 6), fontsize=8, color=cL)
    axR = axL.twinx()
    axR.plot(np.arange(1, len(cum_energy) + 1)[:r_max], cum_energy[:r_max] * 100,
             color=cR, lw=1.3, ls="--")
    axR.set_ylabel("POD cumulative energy (%)", color=cR)
    axR.tick_params(axis="y", labelcolor=cR)
    axR.set_ylim(75, 100.5)
    axR.spines["top"].set_visible(False)
    save_figure(figA, str(outdir / "lespod_conditioning"))
    plt.close(figA)

    # ══ Figure B+C: pressure-augmented observability ══
    figBC, (axB, axC) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    # (a) singular spectra
    cu, mu, lu = STYLE_CYCLE[0]
    cp, mp, lp = STYLE_CYCLE[1]
    axB.semilogy(np.arange(1, len(s_u) + 1), s_u / s_u[0], color=cu, ls=lu,
                 lw=1.5, label="velocity-only")
    axB.semilogy(np.arange(1, len(s_up_rep) + 1), s_up_rep / s_up_rep[0], color=cp, ls=lp,
                 lw=1.5, label="velocity+pressure")
    axB.axvspan(len(s_u), len(s_up_rep), color="0.85", alpha=0.6, zorder=0)
    axB.text((len(s_u) + len(s_up_rep)) / 2, (s_up_rep / s_up_rep[0])[len(s_u)] * 3,
             "pressure-\nrevealed", fontsize=7, ha="center", color="0.35")
    axB.set_xlabel(r"Singular value index $i$")
    axB.set_ylabel(r"$\sigma_i/\sigma_1$ (–)")
    axB.legend(loc="lower left", fontsize=8)
    axB.text(-0.18, 1.02, "(a)", transform=axB.transAxes, fontweight="bold")
    # (b) κ over snapshots
    axC.scatter(np.arange(len(kappa_up)), kappa_up, color=cp, marker=mp, s=28, zorder=3)
    axC.axhline(kappa_u, color=cu, ls=lu, lw=1.3, label=r"velocity-only $\kappa$")
    axC.set_yscale("log")
    axC.set_xlabel(r"LES reference snapshot $\bar a^{(m)}$")
    axC.set_ylabel(r"$\kappa(C_{u,p}^{\mathrm{lin}})$ (–)")
    axC.legend(loc="upper right", fontsize=8)
    axC.text(0.5, 0.06, r"rank $=300$ for all $\bar a$", transform=axC.transAxes,
             fontsize=8, ha="center", color="0.35")
    axC.text(-0.20, 1.02, "(b)", transform=axC.transAxes, fontweight="bold")
    figBC.tight_layout()
    save_figure(figBC, str(outdir / "pressure_observability"))
    plt.close(figBC)

    print(f"[out] {outdir}/lespod_conditioning.pdf  +  pressure_observability.pdf")


if __name__ == "__main__":
    main()

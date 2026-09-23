#!/usr/bin/env python3
"""thesis 三張以 series.npz 為資料源的圖（NTHU thesis 樣式）。

產出：
    main_trajectories.pdf            5-seed 軌跡：KE(t)、div ratio(t)、u/v rel-L2(t)
    band_energy_rel_error_vs_time.pdf 低/中/高頻帶的相對誤差 vs 時間（5-seed）
    kf_mode_diagnostic.pdf           forcing mode 的振幅比與相位誤差（5-seed）

資料源是 evaluate_exp245.py --export-arrays 產出的 series.npz，DNS 對照值已在
檔內（ke_dns、kf_mode_*_dns），故不需要 DNS 場，本機即可繪製。需要場的那批
（field/vorticity/spectrum/mean-profile/temporal）另見 plot_thesis_field_figures.py。

kf_mode 的相位誤差用 wrap 到 (-π, π] 的差值：兩個角度相減後不 wrap，會在
±π 附近跳出 2π 的假尖峰，看起來像相位崩潰。

⚠️ band_energy 圖畫的是 **k_η-fraction、shell-wise** 的 band（`band_rel_err_*`），
不是固定整數 band（k≤5 / 5<k≤16 / k>16）的**積分**量（`low_band_rel_err`）。
thesis 的 caption 與 §3.4 `para:band_conventions` 已按 k_η 約定寫；**TMLR 稿的
`fig:band_err` caption 用的是固定整數 band**，把本腳本的輸出 `--out` 指到
`paper/pof-format/` 會在同一段 caption 底下換掉曲線的定義。要那麼做之前先改
caption。兩個量不可互比。

Usage:
    uv run python scripts/plot_thesis_series_figures.py
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import (  # noqa: E402
    setup_style, figwidth, save_figure, DNS, PICON, DNS_LS, PICON_LS, MUTED,
    STYLE_CYCLE, K_COLORS,
)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = (42, 1, 2, 3, 4)


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _budget_of_run(run_dir: str) -> int | None:
    """從 configs/ 反查某個 artifacts run 目錄的真實 sensor 預算。

    權威鏈是 config 自己：`artifacts_dir` 綁定 run 目錄，`sensor_jsons` 的檔名帶
    `K<N>`。series.npz 裡沒有任何欄位記得預算（k_cut 是 DNS 導出的，三種預算完全
    相同），所以只能回頭問訓練當時用的 config。查不到回 None。
    """
    for cfg in sorted(CONFIG_DIR.glob("*.toml")):
        text = cfg.read_text()
        m = re.search(r'^\s*artifacts_dir\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if not m or Path(m.group(1)).name != run_dir:
            continue
        s = re.search(r"sensors_\w*?_?K(\d+)_", text)
        if s:
            return int(s.group(1))
    return None


def _verify_budget(root: Path, run_dirs: list[str], declared: int) -> None:
    """圖上標的 K 必須等於資料的 K，否則中止。

    Why：`--run-stem`（資料源）與 `--sweep-k`（圖上的標籤與取樣邊界
    sqrt(K/pi)）是兩個獨立旗標，配錯不會有任何徵兆——圖照產、build 全綠、
    caption 照舊。2026-09-18 實際踩過：三張 caption 寫 K=200 的圖是用預設的
    main5_s（K=100）畫的，靠比對正文九個 band 中位數才發現。
    """
    strict = os.environ.get("PILNN_FIGURE_BUDGET_STRICT") == "1"
    unknown = []
    for d in run_dirs:
        actual = _budget_of_run(d)
        if actual is None:
            unknown.append(d)
        elif actual != declared:
            raise SystemExit(
                f"[budget] {root / d} 的 sensor 預算是 K={actual}，"
                f"但 --sweep-k 宣告 K={declared}。\n"
                f"         圖上的取樣邊界與 caption 會與資料不符。"
                f"請改 --run-stem 或改 --sweep-k。"
            )
    if unknown:
        msg = (f"[budget] 無法驗證 {unknown} 的 sensor 預算"
               f"（configs/ 裡沒有 artifacts_dir 指向它的 config）；"
               f"宣告的 K={declared} 未經核對。")
        if strict:
            raise SystemExit(msg + " PILNN_FIGURE_BUDGET_STRICT=1 下視為失敗。")
        print("WARNING: " + msg)


def _load(root: Path, stem: str, subdir: str, seeds=SEEDS) -> list[dict]:
    # stem / subdir 刻意無預設：兩者一起決定讀哪個 sensor 預算的資料，
    # 留一組預設等於再開一次「照預設跑就換掉資料源」的門。
    out = []
    for s in seeds:
        p = root / f"{stem}{s}" / subdir / "series.npz"
        if not p.is_file():
            raise FileNotFoundError(f"缺 seed {s} 的 series.npz: {p}")
        out.append({k: v for k, v in np.load(p).items()})
    lens = {d["t"].size for d in out}
    if len(lens) != 1:
        raise ValueError(f"各 seed 的時間點數不一致: {sorted(lens)}；不可混畫")
    return out


def _envelope(ax, t, curves, colour, ls, label):
    """mean ± 1σ 帶 + 個別 seed 細線。"""
    arr = np.stack(curves)
    m, sd = arr.mean(0), arr.std(0)
    for c in arr:
        ax.plot(t, c, color=colour, lw=0.4, alpha=0.35)
    ax.fill_between(t, m - sd, m + sd, color=colour, alpha=0.18, lw=0)
    ax.plot(t, m, color=colour, ls=ls, label=label)


def fig_main_trajectories(data: list[dict], out: Path, venue: str = "thesis") -> None:
    t = data[0]["t"]
    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=(figwidth(venue, "single"), 6.2))

    ax = axes[0]
    ax.plot(t, data[0]["ke_dns"], color=DNS, ls=DNS_LS, label="DNS")
    _envelope(ax, t, [d["ke_pred"] for d in data], PICON, PICON_LS, "PI-CON")
    ax.set_ylabel(r"$\mathrm{KE}(t)$ (m$^2$/s$^2$)")
    ax.legend(loc="lower right")
    ax.text(0.01, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    _envelope(ax, t, [100 * d["div_ratio"] for d in data], PICON, PICON_LS, "PI-CON")
    # DNS 有限差分底線：同一條件下 DNS 場自身的離散散度，是可達下界
    floor = 100 * np.mean([d["div_dns_l2"] / (d["div_pred_l2"] / d["div_ratio"])
                           for d in data], axis=0)
    ax.plot(t, floor, color=DNS, ls=":", label="DNS finite-difference floor")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Vert\nabla\!\cdot\!\mathbf{u}\Vert_2 / \Vert\nabla\mathbf{u}\Vert_F^{\rm DNS}$ (%)")
    ax.legend(loc="upper right")
    ax.text(0.01, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    ax = axes[2]
    for i, (key, lab) in enumerate([("u_rel_err", "$u$"), ("v_rel_err", "$v$")]):
        colour, _, ls = STYLE_CYCLE[i]
        _envelope(ax, t, [100 * d[key] for d in data], colour, ls, lab)
    ax.set_ylabel(r"relative $L_2$ error (%)")
    ax.set_xlabel(r"$t$ (s)")
    ax.legend(loc="upper right")
    ax.text(0.01, 1.02, "(c)", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout()
    print(f"[out] {[str(p) for p in save_figure(fig, str(out / 'main_trajectories'))]}")
    plt.close(fig)


#: band_rel_err_* 的邊界定義（metric_artifact._bands）。印在 stdout 讓「這張圖畫的
#: 是哪一套 band」進到 job log，而不是只活在讀者的記憶裡。
BAND_CONVENTION = "k_eta-fraction shell-wise: (0,0.1k_eta] / (0.1,0.4] / (0.4,1.0]"


def fig_band_energy(data: list[dict], out: Path, venue: str = "thesis") -> None:
    print(f"[band] convention = {BAND_CONVENTION}"
          "  (NOT the fixed integer bands k<=5 / 5<k<=16 / k>16)")
    # t=0 的高波段能量近零，相對誤差在該幀發散（正文同樣把它排除在統計之外）。
    # 保留它會把縱軸撐成六個 decade，把正文討論的 1%--30% 區間壓進上三分之一。
    t = data[0]["t"][1:]
    fig, ax = plt.subplots(figsize=(figwidth(venue, "single"), 3.0))
    bands = [("band_rel_err_low", "low band"),
             ("band_rel_err_mid", "mid band"),
             ("band_rel_err_high", "high band")]
    for i, (key, lab) in enumerate(bands):
        if key not in data[0]:
            continue
        colour, _, ls = STYLE_CYCLE[i]
        _envelope(ax, t, [100 * np.asarray(d[key])[1:] for d in data], colour, ls, lab)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"band energy relative error (%)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    print(f"[out] {[str(p) for p in save_figure(fig, str(out / 'band_energy_rel_error_vs_time'))]}")
    plt.close(fig)


def _wrap(a: np.ndarray) -> np.ndarray:
    """把角度差 wrap 到 (-π, π]；不 wrap 會在 ±π 附近造出 2π 的假尖峰。"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def fig_kf_mode(data: list[dict], out: Path, venue: str = "thesis") -> None:
    need = ("kf_mode_amp_pred", "kf_mode_amp_dns",
            "kf_mode_phase_pred", "kf_mode_phase_dns")
    missing = [k for k in need if k not in data[0]]
    if missing:
        raise KeyError(
            f"series.npz 缺 {missing}；該欄位由 forcing_mode_coeff_u 產出，"
            "需用含該功能的 evaluate_exp245.py 重跑 eval")
    t = data[0]["t"]
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=(figwidth(venue, "single"), 4.4))

    ax = axes[0]
    ratio = [d["kf_mode_amp_pred"] / (d["kf_mode_amp_dns"] + 1e-12) for d in data]
    _envelope(ax, t, ratio, PICON, PICON_LS, "PI-CON / DNS")
    ax.axhline(1.0, color=DNS, ls=DNS_LS, lw=0.8)
    ax.set_ylabel(r"$|\hat{u}_{k_f}|_{\rm pred} / |\hat{u}_{k_f}|_{\rm DNS}$")
    ax.legend(loc="lower right")
    ax.text(0.01, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    dphi = [_wrap(d["kf_mode_phase_pred"] - d["kf_mode_phase_dns"]) for d in data]
    _envelope(ax, t, dphi, PICON, PICON_LS, "phase error")
    ax.axhline(0.0, color=DNS, ls=DNS_LS, lw=0.8)
    ax.set_ylabel(r"$\arg\hat{u}_{\rm pred} - \arg\hat{u}_{\rm DNS}$ (rad)")
    ax.set_xlabel(r"$t$ (s)")
    ax.legend(loc="upper right")
    ax.text(0.01, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout()
    print(f"[out] {[str(p) for p in save_figure(fig, str(out / 'kf_mode_diagnostic'))]}")
    plt.close(fig)


def fig_energy_spectrum(data: list[dict], out: Path, K: int = 100,
                        stems: tuple[str, ...] = ("energy_spectrum",
                                                  "spectrum_K100_nyquist"),
                        venue: str = "thesis") -> None:
    """t=5 的徑向能譜，含 forcing 波數與 sensor 取樣帶邊界。

    k_max^sensor = sqrt(K/π) 是 K 個點取樣所能解析的波數上界（面積論證），
    論文用它界定「能量主導低頻帶是否被恢復」。
    """
    E_p = np.stack([d["E_pred_k"][-1] for d in data])   # t=5
    E_d = data[0]["E_dns_k"][-1]
    k = np.arange(E_d.size)
    m, sd = E_p.mean(0), E_p.std(0)

    fig, ax = plt.subplots(figsize=(figwidth(venue, "single"), 3.2))
    ax.loglog(k[1:], E_d[1:], color=DNS, ls=DNS_LS, label="DNS")
    ax.fill_between(k[1:], (m - sd)[1:], (m + sd)[1:], color=PICON, alpha=0.2, lw=0)
    ax.loglog(k[1:], m[1:], color=PICON, ls=PICON_LS, label="PI-CON")

    k_sensor = np.sqrt(K / np.pi)
    ax.axvline(k_sensor, color=MUTED, ls="-.", lw=0.8)
    ax.annotate(rf"$k^{{\rm sensor}}_{{\max}}\approx{k_sensor:.2f}$",
                xy=(k_sensor, ax.get_ylim()[1]), xytext=(k_sensor * 1.15, m[1] * 0.5),
                fontsize=7, color="0.35")
    ax.axvline(2, color=MUTED, ls=":", lw=0.8)
    ax.annotate(r"$k_f=2$", xy=(2, m[2]), xytext=(2 * 0.55, m[1] * 0.02),
                fontsize=7, color="0.35")

    # k^-3：2D 正向 enstrophy cascade（Kraichnan），anchor 在 k_f 之後
    ref = k >= 3
    if ref.sum() > 4:
        anchor = int(np.argmin(np.abs(k - 4)))
        c = E_d[anchor] * (k[anchor] ** 3)
        kk = k[ref][:40]
        ax.loglog(kk, c * kk ** -3.0, color="0.5", ls=(0, (4, 2)), lw=0.8,
                  label=r"$k^{-3}$")

    ax.set_xlabel(r"$k$ (1/m)")
    ax.set_ylabel(r"$E(k)$ (m$^3$/s$^2$)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    for stem in stems:
        print(f"[out] {[str(q) for q in save_figure(fig, str(out / stem))]}")
    plt.close(fig)


def fig_energy_spectrum_multi(bundles: list[tuple[int, list[dict]]], out: Path,
                              stem: str = "spectrum_multiK_nyquist",
                              venue: str = "thesis",
                              floor: float = 1e-14) -> None:
    """單軸疊放多個 sensor 預算的 t=5 徑向能譜。

    取代原本的三聯 subfigure。三格各以 0.32\\linewidth 置入，相對造圖寬是 0.34x，
    標稱 10/9/7 pt 的字落地只剩 3.4/3.1/2.4 pt；而三格共用同一條 DNS 曲線與同一條
    k^-3 參考線，真正要讀的「取樣邊界隨 K 右移」反被拆到跨格比較。單軸疊放後字級回到
    標稱值，該比較也變成同軸比較。

    floor 用來裁掉重建曲線觸底後的數值噪聲段；DNS 參考線不受裁切影響。
    """
    if not bundles:
        raise ValueError("bundles 不可為空")
    Ks = [K for K, _ in bundles]
    if len(set(Ks)) != len(Ks):
        raise ValueError(f"sensor 預算重複: {Ks}")

    E_d = bundles[0][1][0]["E_dns_k"][-1]
    for K, data in bundles[1:]:
        ref = data[0]["E_dns_k"][-1]
        if ref.shape != E_d.shape or not np.allclose(ref, E_d, rtol=1e-8, atol=0.0):
            raise ValueError(f"K={K} 的 DNS 參考譜與第一組不同，不可疊在同一軸上")
    k = np.arange(E_d.size)

    fig, ax = plt.subplots(figsize=(figwidth(venue, "single"), 3.4))
    ax.loglog(k[1:], E_d[1:], color=DNS, ls=DNS_LS, label="DNS", zorder=3)

    for K, data in bundles:
        E_p = np.stack([d["E_pred_k"][-1] for d in data])
        m = E_p.mean(0)
        c = K_COLORS.get(K, PICON)
        ax.loglog(k[1:], m[1:], color=c, ls="--", lw=1.4, label=rf"PI-CON $K={K}$")
        k_sensor = np.sqrt(K / np.pi)
        ax.axvline(k_sensor, color=c, ls="-.", lw=0.8, alpha=0.7)

    ax.axvline(2, color=MUTED, ls=":", lw=0.8)
    ax.annotate(r"$k_f=2$", xy=(2, E_d[2]), xytext=(2 * 0.50, E_d[1] * 3.0),
                fontsize=7, color="0.35")
    ax.annotate(r"$k^{\rm sensor}_{\max}$", xy=(np.sqrt(Ks[0] / np.pi), E_d[1]),
                xytext=(np.sqrt(min(Ks) / np.pi) * 1.05, E_d[1] * 3.0),
                fontsize=7, color="0.35")

    # k^-3：2D 正向 enstrophy cascade（Kraichnan），anchor 在 k_f 之後
    ref = k >= 3
    if ref.sum() > 4:
        anchor = int(np.argmin(np.abs(k - 4)))
        c3 = E_d[anchor] * (k[anchor] ** 3)
        kk = k[ref][:40]
        ax.loglog(kk, c3 * kk ** -3.0, color="0.5", ls=(0, (4, 2)), lw=0.8,
                  label=r"$k^{-3}$")

    ax.set_ylim(bottom=floor)
    ax.set_xlabel(r"$k$ (1/m)")
    ax.set_ylabel(r"$E(k)$ (m$^3$/s$^2$)")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    print(f"[out] {[str(q) for q in save_figure(fig, str(out / stem))]}")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parent.parent
    p.add_argument("--artifacts-root", default=str(repo / "artifacts" / "kolmogorov"))
    # 預設＝thesis 收錄的那批。三張圖的 caption 都寫 K=200（chapter04 fig:main_trajectories /
    # fig:band_err、appendix06 fig:kf_mode），而輸出檔名不帶 K，照預設跑一次就會就地覆蓋。
    p.add_argument("--run-stem", default="ksweep_k200_s",
                   help="artifacts 目錄前綴，seed 接在後面（K=100 那批是 main5_s）")
    p.add_argument("--sweep-k", type=int, default=200,
                   help="本批資料的 sensor 數；決定圖上的取樣邊界與 spectrum_K*_nyquist 檔名。"
                        "與 --run-stem 的真實預算不符會中止")
    p.add_argument("--eval-subdir", default="eval_figs",
                   help="讀哪個評估子目錄（帶 --export-arrays 的那次）；"
                        "K=100 的 main5_s 那批在 final_eval")
    p.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                        / "figures" / "results"))
    p.add_argument("--venue", default="thesis",
                   help="journal_style venue：決定字體、圖寬與框線樣式（thesis / tmlr）")
    p.add_argument("--multi-k", default=None,
                   help="併圖模式：以 'K:相對路徑' 逗號分隔，路徑指向含 series.npz 的目錄，"
                        "可用 {seed} 佔位。例如 '100:main5_s{seed}/final_eval,"
                        "200:ksweep_k200_s{seed}/eval_figs,400:ksweep_K400_b3/final_eval'。"
                        "給了就只產單軸疊放的 spectrum_multiK_nyquist，不產三聯圖的分格。")
    p.add_argument("--multi-seeds", default="42",
                   help="--multi-k 展開 {seed} 用的 seed 清單（逗號分隔）。"
                        "原三聯圖是單 seed 42，預設沿用以維持可比性。")
    args = p.parse_args()

    setup_style(args.venue)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.multi_k:
        seeds = [int(x) for x in args.multi_seeds.split(",") if x.strip()]
        root = Path(args.artifacts_root)
        bundles = []
        for spec in args.multi_k.split(","):
            kk, _, rel = spec.partition(":")
            if not rel:
                raise SystemExit(f"--multi-k 格式應為 'K:相對路徑'，收到 {spec!r}")
            paths = ([root / rel.format(seed=s) for s in seeds]
                     if "{seed}" in rel else [root / rel])
            data = []
            for q in paths:
                f = q / "series.npz"
                if not f.is_file():
                    raise SystemExit(f"K={kk} 缺 series.npz: {f}")
                data.append({k: v for k, v in np.load(f).items()})
            lens = {d["t"].size for d in data}
            if len(lens) != 1:
                raise SystemExit(f"K={kk} 各 seed 時間點數不一致: {sorted(lens)}")
            _verify_budget(root, [Path(rel.format(seed=seeds[0])).parts[0]], int(kk))
            bundles.append((int(kk), data))
        fig_energy_spectrum_multi(bundles, out, venue=args.venue)
        print(f"[data] multi-K {[(K, len(d)) for K, d in bundles]} seeds={seeds}")
        return 0

    root = Path(args.artifacts_root)
    _verify_budget(root, [f"{args.run_stem}{s}" for s in SEEDS], args.sweep_k)
    data = _load(root, stem=args.run_stem, subdir=args.eval_subdir)
    fig_main_trajectories(data, out, venue=args.venue)
    fig_band_energy(data, out, venue=args.venue)
    fig_kf_mode(data, out, venue=args.venue)
    # 三聯圖 spectrum_K{100,200,400}_nyquist 的每一格必須用該 K 自己的資料——
    # 取樣邊界 sqrt(K/pi) 會畫在圖上，用錯 K 的資料等於畫錯的邊界。
    fig_energy_spectrum(data, out, K=args.sweep_k,
                        stems=("energy_spectrum", f"spectrum_K{args.sweep_k}_nyquist"),
                        venue=args.venue)
    print(f"[data] {len(data)} seeds, T={data[0]['t'].size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

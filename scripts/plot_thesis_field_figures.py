#!/usr/bin/env python3
"""thesis 四張需要 DNS 場的圖（NTHU thesis 樣式）。

產出（皆為 seed 42 的單一 realization，與 thesis caption 一致）：
    field_comparison_t5.pdf      t=5 的 u/v/ω：DNS｜PI-CON｜絕對誤差
    vorticity_comparison_t5.pdf  t=5 渦量：DNS｜PI-CON｜帶號誤差
    mean_profile_reynolds.pdf    x-平均速度剖面與 Reynolds 應力 <u'v'>(y)
    temporal_consistency.pdf     單點探針時間軌跡與其功率譜

資料源：由 --fields 指定的 fields.npz（evaluate_exp245 --export-fields）＋ 對應的
DNS 場。DNS 路徑取自 fields.npz 內記錄的 dns_path，不另外猜——配錯 DNS 會
畫出看似合理但比錯對象的圖。

場圖的軸樣式與線圖不同（需四邊完整框線標示空間域邊界、不要格線），故各場
panel 另行覆寫，不改全域 rcParams。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import setup_style, figwidth, save_figure, DNS, PICON, DNS_LS, PICON_LS  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pi_lnn_jax.data import load_dns_from_path  # noqa: E402
from pi_lnn_jax.metric_artifact import compute_vorticity  # noqa: E402


def _style_field_ax(ax, with_labels: bool = True) -> None:
    """場 panel：四邊框標示域邊界、無格線、SI 單位軸標。"""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
    ax.grid(False)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=7)
    if with_labels:
        ax.set_xlabel(r"$x$ (m)", fontsize=8, labelpad=2)
        ax.set_ylabel(r"$y$ (m)", fontsize=8, labelpad=2)


def _imshow(ax, f, vmin=None, vmax=None, cmap="RdBu_r"):
    # 場慣例 f[x, y] → imshow 需要 [row=y, col=x]，故轉置；origin=lower 讓 y 向上
    return ax.imshow(np.asarray(f).T, origin="lower", extent=(0, 1, 0, 1),
                     vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")


def _sym(*arrs) -> float:
    return float(max(np.abs(np.asarray(a)).max() for a in arrs))


def fig_field_comparison(u_p, v_p, u_d, v_d, w_p, w_d, out: Path,
                         venue: str = "thesis") -> None:
    rows = [(r"$u$ [m/s]", u_d, u_p), (r"$v$ [m/s]", v_d, v_p),
            (r"$\omega$ [1/s]", w_d, w_p)]
    # aspect=equal 的 3x3 場陣列：高度由 panel 寬度決定，給太高只會產生列間空白
    fig, axes = plt.subplots(3, 3, figsize=(figwidth(venue, "single"), 5.0),
                             constrained_layout=True)
    for r, (lab, d, p) in enumerate(rows):
        lim = _sym(d, p)
        for c, (title, f, kw) in enumerate([
                ("DNS", d, dict(vmin=-lim, vmax=lim)),
                ("PI-CON", p, dict(vmin=-lim, vmax=lim)),
                ("absolute error", np.abs(p - d), dict(vmin=0, vmax=None, cmap="magma"))]):
            ax = axes[r, c]
            im = _imshow(ax, f, **kw)
            # 左欄的量名由 row label 表達，故不再重複 ylabel（兩者會疊在一起）
            _style_field_ax(ax, with_labels=False)
            if r == 2:
                ax.set_xlabel(r"$x$ (m)", fontsize=8, labelpad=2)
            if r == 0:
                ax.set_title(title, fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8, labelpad=2)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    print(f"[out] {[str(q) for q in save_figure(fig, str(out / 'field_comparison_t5'))]}")
    plt.close(fig)


def fig_vorticity(w_p, w_d, out: Path, venue: str = "thesis") -> None:
    lim = _sym(w_d, w_p)
    err = w_p - w_d
    elim = _sym(err)
    fig, axes = plt.subplots(1, 3, figsize=(figwidth(venue, "single"), 2.5))
    for ax, (title, f, kw) in zip(axes, [
            ("DNS", w_d, dict(vmin=-lim, vmax=lim)),
            ("PI-CON", w_p, dict(vmin=-lim, vmax=lim)),
            ("signed error", err, dict(vmin=-elim, vmax=elim))]):
        im = _imshow(ax, f, **kw)
        _style_field_ax(ax, with_labels=(title == "DNS"))
        ax.set_title(title, fontsize=8, pad=3)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    fig.tight_layout()
    print(f"[out] {[str(q) for q in save_figure(fig, str(out / 'vorticity_comparison_t5'))]}")
    plt.close(fig)


def fig_mean_profile(u_p, v_p, u_d, v_d, y, out: Path,
                     venue: str = "thesis") -> None:
    """x-與-t 平均剖面與 Reynolds 應力；輸入為 [T,Nx,Ny]（已切到平均窗）。

    thesis caption 宣告的是 post-spin-up 窗 t∈[1,5] 的時間平均，不是 t=5
    的瞬時剖面——瞬時剖面帶有單一快照的湍流漲落，兩者不是同一個量。
    脈動量 u' 亦對 (x,t) 取平均後才算 Reynolds 應力。
    """
    fig, axes = plt.subplots(1, 2, figsize=(figwidth(venue, "single"), 2.8))
    # 對 (t, x) 兩軸平均 → 剩 y
    ax = axes[0]
    ax.plot(u_d.mean(axis=(0, 1)), y, color=DNS, ls=DNS_LS, label="DNS")
    ax.plot(u_p.mean(axis=(0, 1)), y, color=PICON, ls=PICON_LS, label="PI-CON")
    ax.set_xlabel(r"$\langle u\rangle_{x,t}$ (m/s)")
    ax.set_ylabel(r"$y$ (m)")
    ax.legend(loc="upper right")
    ax.text(0.02, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    for f_u, f_v, colour, ls, lab in [(u_d, v_d, DNS, DNS_LS, "DNS"),
                                      (u_p, v_p, PICON, PICON_LS, "PI-CON")]:
        up = f_u - f_u.mean(axis=1, keepdims=True)
        vp = f_v - f_v.mean(axis=1, keepdims=True)
        ax.plot((up * vp).mean(axis=(0, 1)), y, color=colour, ls=ls, label=lab)
    ax.set_xlabel(r"$\langle u'v'\rangle_{x,t}$ (m$^2$/s$^2$)")
    ax.set_ylabel(r"$y$ (m)")
    ax.legend(loc="upper right")
    ax.text(0.02, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    print(f"[out] {[str(q) for q in save_figure(fig, str(out / 'mean_profile_reynolds'))]}")
    plt.close(fig)


def _spatial_acf(U):
    """逐點時間自相關後對所有格點平均。U: (T, Ny, Nx) -> (T,)

    Why 空間平均而非單點：單一探針是一次實現，其 1/e 時間受局部渦結構主導
    （實測中心探針 0.25 s vs 場平均 0.37 s）。caption 宣告的是場的統計量。
    """
    T = U.shape[0]
    X = U.reshape(T, -1).astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    n = 1 << int(np.ceil(np.log2(2 * T)))
    F = np.fft.rfft(X, n=n, axis=0)
    a = np.fft.irfft(F * np.conj(F), n=n, axis=0)[:T]
    a /= a[0:1]
    return a.mean(1)


def _t_1e(r, dt):
    """R 首次跌破 1/e 的時刻，對兩個相鄰樣本線性內插。"""
    below = np.where(r < 1.0 / np.e)[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return 0.0
    frac = (r[i - 1] - 1.0 / np.e) / (r[i - 1] - r[i])
    return (i - 1 + frac) * dt


def fig_temporal(u_pred, dns_u, t, out: Path, les_probe=None,
                 venue: str = "thesis") -> None:
    r"""(a) 域中心探針軌跡；(b) 空間平均的時間自相關 $R_{uu}(\tau)$。

    (b) 在 post-spin-up 窗 $t\ge1$ s 上計算，與本章其餘時間診斷同窗；
    先逐點算時間自相關再對格點平均，所以它是場的統計量、不是單點實現。
    """
    c = u_pred.shape[1] // 2
    u_p_t, u_d_t = u_pred[:, c, c], dns_u[:, c, c]
    fig, axes = plt.subplots(1, 2, figsize=(figwidth(venue, "single"), 2.8))

    ax = axes[0]
    ax.plot(t, u_d_t, color=DNS, ls=DNS_LS, label="DNS")
    ax.plot(t, u_p_t, color=PICON, ls=PICON_LS, label="PI-CON")
    if les_probe is not None:
        lt, lu = les_probe
        ax.plot(lt, lu, color="0.45", ls=":", lw=1.1, label="forward LES")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$u(\mathbf{x}_0,t)$ (m/s)")
    ax.legend(loc="upper right", fontsize="x-small")
    ax.text(0.02, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    win = np.asarray(t) >= 1.0
    dt = float(np.mean(np.diff(np.asarray(t)[win])))
    for U, colour, ls, lab in [(dns_u, DNS, DNS_LS, "DNS"),
                               (u_pred, PICON, PICON_LS, "PI-CON")]:
        r = _spatial_acf(np.asarray(U)[win])
        lag = np.arange(r.size) * dt
        keep = lag <= 1.5
        ax.plot(lag[keep], r[keep], color=colour, ls=ls, label=lab)
        print(f"[acf] {lab}: 1/e time = {_t_1e(r, dt):.4f} s "
              f"(spatially averaged, t>=1 s, n={int(win.sum())} frames)")
    ax.axhline(1.0 / np.e, color="0.5", ls=":", lw=0.8)
    ax.text(0.98, 1.0 / np.e, r"$1/e$", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize="x-small", color="0.4")
    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel(r"$R_{uu}(\tau)$")
    ax.legend(loc="upper right", fontsize="x-small")
    ax.text(0.02, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    print(f"[out] {[str(q) for q in save_figure(fig, str(out / 'temporal_consistency'))]}")
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", default=str(repo / "artifacts" / "kolmogorov"
                                           / "main5_s42" / "final_eval" / "fields.npz"))
    p.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                        / "figures" / "results"))
    p.add_argument("--les-probe", default=None,
                   help="forward LES（DNS 初始化，同 IC）的中心探針 npz，含 t 與 u。\n"
                        "來源：home-gpu ~/les-gen/output/kolmogorov_les_Re10000_N256_T5_dns_init_FIXED.npy\n"
                        "的中心點時序；不給則 (a) 面板不畫該曲線。")
    p.add_argument("--venue", default="thesis",
                   help="journal_style venue：決定字體、圖寬與框線樣式（thesis / tmlr）")
    args = p.parse_args()

    fp = Path(args.fields)
    if not fp.is_file():
        raise FileNotFoundError(f"缺 fields.npz: {fp}（需 evaluate_exp245 --export-fields）")
    F = np.load(fp)
    u_pred, v_pred, t = F["u_pred"], F["v_pred"], F["t"]
    y = F["y"]

    # DNS 路徑取自 fields.npz 記錄，不猜——配錯 DNS 會畫出看似合理卻比錯對象的圖
    dns_path = str(F["dns_path"])
    dns_u, dns_v, dns_t = load_dns_from_path(dns_path, time_stride=1)
    if dns_u.shape[0] != u_pred.shape[0]:
        # eval 端用 --time-stride 取樣過；用同樣的 stride 對齊，不做寬鬆截短
        stride = (dns_u.shape[0] - 1) // (u_pred.shape[0] - 1)
        dns_u, dns_v, dns_t = dns_u[::stride], dns_v[::stride], dns_t[::stride]
    if dns_u.shape[0] != u_pred.shape[0]:
        raise ValueError(f"DNS {dns_u.shape[0]} 幀無法對齊 pred {u_pred.shape[0]} 幀")
    if not np.allclose(dns_t, t, rtol=1e-4, atol=1e-6):
        raise ValueError(f"時間軸不符: dns[:3]={dns_t[:3]} fields[:3]={t[:3]}")

    les_probe = None
    if args.les_probe:
        lp = Path(args.les_probe)
        if not lp.is_file():
            raise FileNotFoundError(f"缺 les probe: {lp}")
        Lz = np.load(lp)
        les_probe = (np.asarray(Lz["t"]), np.asarray(Lz["u"]))

    setup_style(args.venue)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    i5 = int(np.argmin(np.abs(t - 5.0)))
    u_p, v_p, u_d, v_d = u_pred[i5], v_pred[i5], dns_u[i5], dns_v[i5]
    w_p = compute_vorticity(u_p, v_p)
    w_d = compute_vorticity(u_d, v_d)
    print(f"[data] t={float(t[i5]):.2f}s (index {i5}/{t.size - 1})")

    fig_field_comparison(u_p, v_p, u_d, v_d, w_p, w_d, out, venue=args.venue)
    fig_vorticity(w_p, w_d, out, venue=args.venue)
    # mean profile 用 post-spin-up 窗 t∈[1,5]（caption 宣告值），非 t=5 瞬時
    win = t >= 1.0
    fig_mean_profile(u_pred[win], v_pred[win], dns_u[win], dns_v[win], y, out,
                     venue=args.venue)
    fig_temporal(u_pred, dns_u, t, out, les_probe=les_probe, venue=args.venue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

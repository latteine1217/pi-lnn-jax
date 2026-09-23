"""Generate a deterministic PI-CON DeepONet architecture figure (TMLR / pi-lnn-jax).

What: 產生論文用兩行 DeepONet branch/trunk 架構圖，輸出到 paper/tmlr-format/figures/architecture。
Why: Image generation 容易新增錯誤箭頭；固定座標與箭頭拓撲可避免
     Branch Basis / Trunk Basis 被誤接到 Field。
Note: 移植自 pi-lnn (PyTorch) thesis script。liquid-τ 併入 Liquid CfC Memory box
     呈現（CfC 的一部分，沿用 badge 1，不另標注）；並修正若干 subtitle 文字溢出。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper/tmlr-format" / "figures"
OUT_STEM = OUT_DIR / "architecture"

COLORS = {
    "ink": "#1f2933",
    "muted": "#59636f",
    "blue": "#2567ad",
    "blue_fill": "#eaf3ff",
    "teal": "#16817a",
    "teal_fill": "#e8f7f4",
    "purple": "#6d46b5",
    "purple_fill": "#f0eafd",
    "orange": "#d96c00",
    "orange_fill": "#fff4e5",
    "rust": "#b43d15",
    "rust_fill": "#fff0e8",
    "gray": "#374151",
    "gray_fill": "#eef2f7",
    "line": "#253142",
    "light": "#cbd5df",
    # 貢獻標示：單一 accent（this work 的 3 個新元件）。灰底骨幹 + accent novelty，
    # 並以粗框 + 編號 badge 作為非顏色通道，確保灰階列印仍可區分。
    "novel": "#1f6fb2",
    "novel_fill": "#e7f1fb",
    "backbone": "#566173",
    "backbone_fill": "#eef2f7",
}


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    *,
    edge: str,
    face: str,
    title_color: str | None = None,
    title_size: float = 8.2,
    subtitle_size: float = 6.5,
    lw: float = 1.35,
) -> None:
    """固定尺寸節點，文字置中，避免自動 layout 改變拓撲。"""
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.075",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * (0.62 if subtitle else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color or edge,
        zorder=4,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.33,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["ink"],
            linespacing=1.25,
            zorder=4,
        )


def add_chip(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    edge: str,
    width: float,
    height: float = 0.34,
    fontsize: float = 6.5,
) -> None:
    chip = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.0,
        edgecolor=edge,
        facecolor="white",
        zorder=5,
    )
    ax.add_patch(chip)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=edge,
        zorder=6,
    )


def add_badge(ax, cx: float, cy: float, num: int, *, color: str = COLORS["novel"], r: float = 0.17) -> None:
    """貢獻編號 badge：自繪實心圓 + 白色數字，避免依賴字型的 circled-digit glyph。"""
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=color, edgecolor="white", linewidth=1.1, zorder=7))
    ax.text(cx, cy, str(num), ha="center", va="center", fontsize=8.0, fontweight="bold", color="white", zorder=8)


def add_legend(ax, x: float, y: float) -> None:
    """灰底骨幹 vs accent 貢獻的雙列圖例，呼應 tab:deeponet_gaps 的 1/2/3 編號。"""
    sw = 0.40
    h = 0.26
    # 第一列：inherited backbone
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), sw, h, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=COLORS["backbone_fill"], edgecolor=COLORS["backbone"], linewidth=1.0, zorder=5,
        )
    )
    ax.text(x + sw + 0.16, y + h / 2, "Inherited DeepONet backbone", fontsize=6.8, va="center", color=COLORS["ink"], zorder=6)
    # 第二列：this work 的貢獻
    y2 = y - 0.46
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y2), sw, h, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=COLORS["novel_fill"], edgecolor=COLORS["novel"], linewidth=2.2, zorder=5,
        )
    )
    add_badge(ax, x + sw + 0.16, y2 + h / 2, 1, r=0.135)
    ax.text(
        x + sw + 0.40, y2 + h / 2,
        "Added by PI-CON (numbered 1-3)",
        fontsize=6.8, va="center", color=COLORS["novel"], zorder=6, fontweight="bold",
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    lw: float = 1.35,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    mutation_scale: float = 12.0,
    zorder: int = 2,
) -> None:
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=5,
        shrinkB=5,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label:
        if label_xy is None:
            label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=5.9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92),
            zorder=6,
        )


def add_bracket(ax, x0: float, x1: float, y: float, label: str) -> None:
    """畫 trainable modules bracket，避免 feedback 指到 observations。"""
    path = MplPath(
        [(x0, y), (x0, y + 0.16), (x1, y + 0.16), (x1, y)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
    )
    ax.add_patch(
        patches.PathPatch(
            path,
            fill=False,
            edgecolor=COLORS["gray"],
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        y + 0.28,
        label,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=COLORS["gray"],
        zorder=5,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(12.4, 5.8), dpi=240)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    # 分區標題（zone label）：統一中性灰，讓 accent 顏色專門表達「貢獻」
    ax.text(0.35, 6.25, "Branch path: sparse sensor memory", fontsize=8.5, fontweight="bold", color=COLORS["muted"])
    ax.text(2.55, 2.95, "Trunk path: query feature", fontsize=8.0, fontweight="bold", color=COLORS["muted"])
    ax.text(7.05, 6.25, "Interaction", fontsize=8.0, fontweight="bold", color=COLORS["muted"])
    ax.text(11.35, 6.25, "Output and loss", fontsize=8.0, fontweight="bold", color=COLORS["muted"])

    # Branch row
    obs = (0.35, 4.55, 1.72, 1.08)
    spatial = (2.62, 4.55, 1.82, 1.08)
    cfc = (4.95, 4.55, 2.02, 1.08)
    add_box(ax, *obs, "Observations", "sensor values [T,K,{u,v}]\npositions + time", edge=COLORS["backbone"], face=COLORS["backbone_fill"])
    add_chip(ax, obs[0] + 0.25, obs[1] + obs[3] + 0.09, "K = 100 sensors", edge=COLORS["backbone"], width=1.22)
    add_box(ax, *spatial, "Spatial Tokens", "Fourier/RFF encoding\nresidual MLP", edge=COLORS["backbone"], face=COLORS["backbone_fill"])
    add_chip(ax, spatial[0] + 0.26, spatial[1] + spatial[3] + 0.09, "tokens: [T,K,d]", edge=COLORS["backbone"], width=1.30)
    add_box(ax, *cfc, "Liquid CfC", "token attention, causal scan\ninput-dependent tau", edge=COLORS["novel"], face=COLORS["novel_fill"], lw=2.4, title_size=8.0, subtitle_size=6.1)
    add_chip(ax, cfc[0] + 0.52, cfc[1] + cfc[3] + 0.09, "h: [T,K,d]", edge=COLORS["novel"], width=1.00)
    add_badge(ax, cfc[0] + 0.16, cfc[1] + cfc[3] - 0.16, 1)
    add_arrow(ax, (obs[0] + obs[2], 5.12), (spatial[0], 5.12), label="encode", label_xy=(2.36, 5.35))
    add_arrow(ax, (spatial[0] + spatial[2], 5.12), (cfc[0], 5.12), label="scan dt", label_xy=(4.66, 5.35))

    # Trunk row
    query = (2.75, 1.92, 1.45, 0.76)
    trunk_feature = (4.95, 1.82, 1.86, 0.94)
    add_box(ax, *query, "Query", "(x, y, t_q, c)", edge=COLORS["backbone"], face=COLORS["backbone_fill"], title_size=8.0)
    add_box(
        ax,
        *trunk_feature,
        "Trunk Feature",
        "Fourier + temporal anchor\ndt_to_query",
        edge=COLORS["backbone"],
        face=COLORS["backbone_fill"],
        title_size=7.6,
        subtitle_size=6.1,
    )
    add_arrow(ax, (query[0] + query[2], 2.28), (trunk_feature[0], 2.28), label="embed", label_xy=(4.56, 2.50))

    # Cross-attention block
    attn = (7.25, 3.76, 2.55, 1.18)
    add_box(
        ax,
        *attn,
        "Cross-Attention Readout",
        "Q from trunk feature\nK,V from CfC memory\n+ isotropic distance bias",
        edge=COLORS["novel"],
        face=COLORS["novel_fill"],
        title_color=COLORS["novel"],
        title_size=7.4,
        subtitle_size=5.9,
        lw=2.4,
    )
    add_badge(ax, attn[0] + 0.16, attn[1] + attn[3] - 0.16, 2)
    add_arrow(
        ax,
        (cfc[0] + cfc[2], 5.12),
        (attn[0], 4.55),
        label="K,V tokens\nlatest h(t_k <= t_q)",
        label_xy=(7.30, 5.34),
        connectionstyle="arc3,rad=-0.08",
    )
    add_arrow(
        ax,
        (trunk_feature[0] + trunk_feature[2], 2.28),
        (attn[0], 4.05),
        label="Q",
        label_xy=(7.05, 3.08),
        connectionstyle="angle3,angleA=0,angleB=-90",
    )

    # Basis + fusion: 嚴格 Y-shaped merge，basis 只進 Fusion。
    branch_basis = (10.35, 4.70, 1.35, 0.64)
    trunk_basis = (10.35, 2.48, 1.35, 0.64)
    fusion = (12.15, 3.48, 1.72, 0.88)
    field = (14.25, 3.52, 1.10, 0.78)
    loss = (14.05, 1.65, 1.92, 1.24)
    add_box(ax, *branch_basis, "Branch Basis", "query-conditioned\ncontext", edge=COLORS["backbone"], face="white", title_size=7.2, subtitle_size=5.6)
    add_box(ax, *trunk_basis, "Trunk Basis", "component-wise\nbasis", edge=COLORS["backbone"], face="white", title_size=7.2, subtitle_size=5.6)
    add_box(ax, *fusion, "DeepONet Fusion", "branch x trunk basis\ncomponent-wise", edge=COLORS["backbone"], face=COLORS["backbone_fill"], title_size=7.2, subtitle_size=5.4)
    add_box(ax, *field, "Field", "u, v, p\n(p: physics-only)", edge=COLORS["backbone"], face="white", title_size=7.2, subtitle_size=5.3)
    add_box(
        ax,
        *loss,
        "Loss",
        "div(u)=0 via AL constraint\nsensor MSE + NS residual\nGradNorm balancing",
        edge=COLORS["novel"],
        face=COLORS["novel_fill"],
        title_color=COLORS["novel"],
        title_size=7.4,
        subtitle_size=5.3,
        lw=2.4,
    )
    add_badge(ax, loss[0] + 0.16, loss[1] + loss[3] - 0.16, 3)
    add_arrow(ax, (attn[0] + attn[2], 4.58), (branch_basis[0], 5.00))
    add_arrow(ax, (trunk_feature[0] + trunk_feature[2], 2.28), (trunk_basis[0], 2.72), connectionstyle="arc3,rad=0.02")
    add_arrow(ax, (branch_basis[0] + branch_basis[2], 5.00), (fusion[0], 4.12), connectionstyle="arc3,rad=-0.06")
    add_arrow(ax, (trunk_basis[0] + trunk_basis[2], 2.72), (fusion[0], 3.72), connectionstyle="arc3,rad=0.06")
    add_arrow(ax, (fusion[0] + fusion[2], 3.92), (field[0], 3.92))
    add_arrow(ax, (field[0] + field[2], 3.92), (loss[0], 2.55), label="autograd", label_xy=(14.35, 3.24), connectionstyle="angle3,angleA=0,angleB=90")

    # Optimizer feedback
    opt = (10.15, 0.72, 1.72, 0.64)
    params = (7.35, 0.72, 1.92, 0.64)
    add_box(ax, *opt, "SOAP + Schedule-Free", "preconditioned updates", edge=COLORS["gray"], face="white", title_color=COLORS["ink"], title_size=6.6, subtitle_size=5.1, lw=1.15)
    add_box(ax, *params, "model parameters theta", "encoder + CfC + decoder", edge=COLORS["gray"], face="white", title_color=COLORS["ink"], title_size=6.5, subtitle_size=5.1, lw=1.15)
    add_arrow(ax, (loss[0] + 0.55, loss[1]), (opt[0] + opt[2], opt[1] + 0.36), label="optimize", label_xy=(13.0, 1.05), linestyle=(0, (4, 3)), lw=1.2)
    add_arrow(ax, (opt[0], opt[1] + 0.36), (params[0] + params[2], params[1] + 0.36), label="update theta", label_xy=(9.82, 1.32), linestyle=(0, (4, 3)), lw=1.2)
    add_bracket(ax, spatial[0] - 0.05, fusion[0] + fusion[2] + 0.05, 1.52, "trainable modules")
    add_arrow(ax, (params[0] + params[2] / 2, params[1] + params[3]), (8.40, 1.68), linestyle=(0, (4, 3)), lw=1.2)

    # Loss formula and supervision note
    ax.text(
        4.05,
        0.23,
        "L = L_data(u,v) + GradNorm(NS_u, NS_v, cont) + AL(cont)",
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.05", fc="white", ec=COLORS["light"], lw=0.9),
        zorder=5,
    )
    ax.text(
        11.2,
        0.23,
        "Full-field DNS is not used as supervision.",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["gray"],
        zorder=5,
    )

    # 圖例：灰底骨幹 vs accent 貢獻（編號 1-3 對應 tab:deeponet_gaps）
    add_legend(ax, 0.40, 1.18)

    fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUT_STEM}.png", bbox_inches="tight", pad_inches=0.04, dpi=360)
    plt.close(fig)


if __name__ == "__main__":
    main()

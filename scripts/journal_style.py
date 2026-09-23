#!/usr/bin/env python3
"""journal_style.py — one-call matplotlib styling for journal figures.

What:
    `setup_style(venue)` applies an rcParams profile tuned to a target venue
    (ICLR/NeurIPS, IEEE, Nature, JFM) so plots come out camera-ready instead of
    matplotlib-default. Also exposes figure-width helpers, a colorblind-safe
    palette, a marker/linestyle cycle, and a `save_figure` that writes vector.

Why:
    Two things sink figures in review: matplotlib defaults (saturated Tab10,
    square markers, top/right spines, 100 DPI) and venue-spec violations (wrong
    width, raster-where-vector-required, sub-minimum fonts). Encoding both here
    means every plot script starts correct rather than re-deriving the numbers.

Usage:
    from journal_style import setup_style, figwidth, PALETTE, STYLE_CYCLE, save_figure

    setup_style("jfm")
    fig, ax = plt.subplots(figsize=(figwidth("jfm", "single"), 2.6))
    for i, (label, y) in enumerate(series.items()):
        c, m, ls = STYLE_CYCLE[i]
        ax.plot(x, y, color=c, marker=m, linestyle=ls, markevery=20, label=label)
    ax.set_xlabel("k"); ax.set_ylabel("E(k)")
    ax.legend()
    save_figure(fig, "thesis/figures/spectrum")   # writes .pdf (+ .png preview)

Design notes:
    - Serif venues (ICLR/IEEE/JFM) get a serif/Computer-Modern label font to
      match the body text; Nature gets sans-serif (Helvetica/Arial).
    - The palette is the Wong colorblind-safe set (Nature Methods 2011). Every
      series also gets a distinct marker + linestyle so it survives grayscale.
    - This module has no project dependencies — copy it next to any plot script.

Two axes of styling, deliberately kept separate:
    - *Presentation* is per-venue. `"thesis"` is boxed (four spines, inward
      ticks, grid, framed legend) because that is the fluids-journal
      convention the NTHU thesis targets; every other venue is minimal
      (no top/right spine, outward ticks, frameless legend) per ML-conference
      convention. Neither is a drift from the other — the repo carries two
      papers with different audiences.
    - *Semantics* is global. DNS/PI-CON/LES and the sweep palettes below are
      shared by both papers so one quantity never changes colour between them.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Figure widths in inches (see references/venues.md) ──────────────────
WIDTHS_IN: dict[str, dict[str, float]] = {
    "thesis":  {"single": 5.906, "half": 2.90},  # NTHU thesis \textwidth = 15 cm (a4, 3 cm margins)
    "iclr":    {"single": 5.50, "half": 2.70},
    "neurips": {"single": 5.50, "half": 2.70},
    "ieee":    {"single": 3.50, "double": 7.16},
    "nature":  {"single": 3.46, "double": 7.20},  # 89 mm / 183 mm
    "jfm":     {"single": 5.00, "half": 2.40},
    "tmlr":    {"single": 6.50, "half": 3.20},  # \textwidth 6.5in (JMLR-style 1-col)
}

# ── Colorblind-safe palette (Wong 2011, Nature Methods) ─────────────────
# Ordered so the first few are maximally distinct. Black anchors references.
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (use sparingly — low contrast on white)
    "#000000",  # black (reference / ground truth)
]

# Marker + linestyle cycle so series differ without relying on color.
# Deliberately avoids the square marker 's' as the primary (user preference);
# 'o'/'^'/'D' read cleaner at small sizes.
_MARKERS = ["o", "^", "D", "v", "s", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
STYLE_CYCLE: list[tuple[str, str, str]] = list(
    zip(PALETTE, _MARKERS, _LINESTYLES)
)

# ── Semantic roles (venue-independent) ──────────────────────────────────
# Single source of truth for what a colour *means*. Import these names rather
# than a raw hex or a PALETTE index, so a quantity keeps its colour across
# both papers and across every figure within one. Colour and line style are
# redundant by design, so figures survive greyscale printing:
#     DNS / ground truth  -> black, solid
#     PI-CON / prediction -> blue,  dashed
#     LES surrogate       -> green, dotted
OKABE_ITO: dict[str, str] = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "green":      "#009E73",  # bluish-green
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",  # reddish-purple
    "grey":       "#999999",
}

DNS      = OKABE_ITO["black"]       # ground truth / reference            (solid)
PICON    = OKABE_ITO["blue"]        # proposed method / prediction        (dashed)
LES      = OKABE_ITO["green"]       # LES statistical surrogate           (dotted)
BASELINE = OKABE_ITO["vermillion"]  # forward-CFD / interpolation baselines
ORACLE   = OKABE_ITO["orange"]      # DNS-oracle placement / secondary reference
OUTPUT   = OKABE_ITO["purple"]      # reconstructed field / output entity (schematics)
ACCENT   = OKABE_ITO["sky"]
MUTED    = OKABE_ITO["grey"]

DNS_LS, PICON_LS, LES_LS = "-", "--", ":"

# Fixed sweep palettes — a given N or K keeps its colour across every figure.
N_COLORS = {128: OKABE_ITO["blue"], 256: OKABE_ITO["vermillion"],
            512: OKABE_ITO["green"], 1024: OKABE_ITO["black"]}
N_MARKERS = {128: "o", 256: "s", 512: "^", 1024: "D"}
N_LINESTYLES = {128: "--", 256: "-", 512: "-.", 1024: ":"}

K_COLORS = {100: OKABE_ITO["blue"], 200: OKABE_ITO["vermillion"],
            400: OKABE_ITO["green"]}

PLACEMENT_COLORS = {"dns": OKABE_ITO["orange"], "les": OKABE_ITO["blue"],
                    "random": OKABE_ITO["vermillion"]}

# Schematic entity fills/edges (method-overview boxes); shares PI-CON blue & LES green
SCHEMATIC = {
    "place_edge": OKABE_ITO["green"],  "place_fill": "#e7f3ec",   # LES / placement
    "dns_edge":   OKABE_ITO["grey"],   "dns_fill":   "#eef1f3",   # DNS reference (neutral)
    "method_edge": OKABE_ITO["blue"],  "method_fill": "#e2eff8",  # PI-CON / inputs
    "out_edge":   OKABE_ITO["purple"], "out_fill":   "#f6ecf2",   # reconstructed field / eval
    "ink":  "#1f2933",
    "line": "#253142",
    "muted": OKABE_ITO["grey"],
}


def figwidth(venue: str, kind: str = "single") -> float:
    """Return the physical figure width in inches for venue/kind.

    kind is 'single'/'half' (single-column venues) or 'single'/'double'
    (two-column venues). Raises on an unknown combination so a typo fails
    loud rather than silently producing a wrong-width figure.
    """
    venue = venue.lower()
    if venue not in WIDTHS_IN:
        raise KeyError(f"unknown venue {venue!r}; known: {sorted(WIDTHS_IN)}")
    table = WIDTHS_IN[venue]
    if kind not in table:
        raise KeyError(
            f"venue {venue!r} has no width {kind!r}; available: {sorted(table)}"
        )
    return table[kind]


# Venues drawn in the boxed fluids-journal presentation (see module docstring).
_BOXED_VENUES = {"thesis"}

# Boxed profile: four spines, inward ticks on all sides, visible grid, framed
# legend. Font sizes are absolute rather than derived from a base, because the
# thesis figures were tuned at these exact sizes against the NTHU body text.
_BOXED_RCPARAMS: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.7,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "grid.color": "#999999",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#666666",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.borderaxespad": 0.4,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    # Embed TrueType; matplotlib's default Type-3 is rejected by some presses.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def setup_style(venue: str = "iclr") -> None:
    """Apply a venue-tuned rcParams profile. Call once before plotting.

    Resets to matplotlib defaults first, so the two presentations stay
    isolated: the boxed profile sets keys (grid, mirrored ticks) that the
    minimal one never mentions, and without a reset those would leak into a
    later `setup_style` call in the same process.
    """
    venue = venue.lower()
    mpl.rcdefaults()
    if venue in _BOXED_VENUES:
        mpl.rcParams.update(_BOXED_RCPARAMS)
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)
        return

    serif = venue in {"iclr", "neurips", "ieee", "jfm", "tmlr"}

    # Per-venue minimum label font at final size (see venues.md).
    base_font = {
        "iclr": 9, "neurips": 9, "ieee": 8, "nature": 7, "jfm": 9, "tmlr": 8,
    }.get(venue, 9)

    if serif:
        family = "serif"
        font_list = ["Times New Roman", "Times", "DejaVu Serif"]
        mathtext = "cm"  # Computer Modern math — matches LaTeX body
    else:  # Nature
        family = "sans-serif"
        font_list = ["Helvetica", "Arial", "DejaVu Sans"]
        mathtext = "dejavusans"

    mpl.rcParams.update({
        "font.family": family,
        f"font.{family}": font_list,
        "mathtext.fontset": mathtext,
        "font.size": base_font,
        "axes.labelsize": base_font,
        "axes.titlesize": base_font + 1,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        # Vector-first: high savefig DPI is a fallback; prefer .pdf output.
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType so fonts are editable/searchable
        "ps.fonttype": 42,
        # Chartjunk removal.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        # Use the colorblind-safe palette as the default color cycle.
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save_figure(fig, stem: str, formats=("pdf", "png")) -> list[Path]:
    """Save `fig` to <stem>.<ext> for each format. Defaults to vector PDF
    plus a PNG preview. Returns the written paths.

    Writing PDF first enforces the vector-first rule; the PNG is a convenience
    preview, not the submission asset.
    """
    out = Path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        p = out.with_suffix(f".{ext}")
        fig.savefig(p)
        written.append(p)
    return written


if __name__ == "__main__":
    # Smoke test: render a tiny multi-series figure in each venue style.
    import numpy as np

    x = np.linspace(0, 2 * np.pi, 200)
    for v in ("thesis", "iclr", "ieee", "nature", "jfm"):
        setup_style(v)
        kind = "double" if v in {"ieee", "nature"} else "single"
        fig, ax = plt.subplots(figsize=(figwidth(v, kind), 2.4))
        for i in range(3):
            c, m, ls = STYLE_CYCLE[i]
            ax.plot(x, np.sin(x + i), color=c, marker=m, linestyle=ls,
                    markevery=25, label=f"series {i}")
        ax.set_xlabel("x"); ax.set_ylabel("amplitude")
        ax.legend()
        paths = save_figure(fig, f"/tmp/journal_style_smoke_{v}")
        print(f"{v}: {[str(p) for p in paths]}")
        plt.close(fig)

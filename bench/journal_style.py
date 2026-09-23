#!/usr/bin/env python3
"""journal_style.py — one-call matplotlib styling for journal figures.

Vendored from the scientific-figures skill and extended with a `tmlr` profile
(TMLR single-column, \\textwidth = 469.76 pt = 6.50 in, serif/Computer-Modern,
8 pt minimum). See references/venues.md in that skill for the other venues.

Usage:
    from journal_style import setup_style, figwidth, PALETTE, STYLE_CYCLE, save_figure
    setup_style("tmlr")
    fig, ax = plt.subplots(figsize=(figwidth("tmlr", "single"), 2.4))
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# ── Figure widths in inches (TMLR measured from tmlr.sty; others per venues.md) ──
WIDTHS_IN: dict[str, dict[str, float]] = {
    "tmlr":    {"single": 6.50, "half": 3.20},  # \textwidth = 469.76 pt
    "iclr":    {"single": 5.50, "half": 2.70},
    "neurips": {"single": 5.50, "half": 2.70},
    "ieee":    {"single": 3.50, "double": 7.16},
    "nature":  {"single": 3.46, "double": 7.20},  # 89 mm / 183 mm
    "jfm":     {"single": 5.00, "half": 2.40},
}

# ── Colorblind-safe palette (Wong 2011, Nature Methods) ─────────────────
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
_MARKERS = ["o", "^", "D", "v", "s", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
STYLE_CYCLE: list[tuple[str, str, str]] = list(zip(PALETTE, _MARKERS, _LINESTYLES))


def figwidth(venue: str, kind: str = "single") -> float:
    """Physical figure width in inches for venue/kind; raises loud on a typo."""
    venue = venue.lower()
    if venue not in WIDTHS_IN:
        raise KeyError(f"unknown venue {venue!r}; known: {sorted(WIDTHS_IN)}")
    table = WIDTHS_IN[venue]
    if kind not in table:
        raise KeyError(f"venue {venue!r} has no width {kind!r}; available: {sorted(table)}")
    return table[kind]


def setup_style(venue: str = "tmlr") -> None:
    """Apply a venue-tuned rcParams profile. Call once before plotting."""
    venue = venue.lower()
    serif = venue in {"tmlr", "iclr", "neurips", "ieee", "jfm"}
    base_font = {
        "tmlr": 8, "iclr": 9, "neurips": 9, "ieee": 8, "nature": 7, "jfm": 9,
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
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save_figure(fig, stem: str, formats=("pdf", "png")) -> list[Path]:
    """Save `fig` to <stem>.<ext> for each format (vector PDF first, PNG preview)."""
    out = Path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        p = out.with_suffix(f".{ext}")
        fig.savefig(p)
        written.append(p)
    return written

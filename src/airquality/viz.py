"""Shared matplotlib style: one validated palette, thin marks, recessive axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from airquality.config import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# Categorical slots in fixed order (CVD-validated; never cycle or reorder).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
THRESHOLD = "#d03b3b"  # reserved status 'critical' - used only for limit lines, always labelled

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (9, 4.2),
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "sans-serif",
    })


def threshold_line(ax, y: float, label: str, x: float = 1.0) -> None:
    """Horizontal reference line for a legal/guideline limit, labelled in text ink.

    `x` is the label's right edge in axes fraction - move it to avoid overlapping data.
    """
    ax.axhline(y, color=THRESHOLD, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, xy=(x, y), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=TEXT_2,
                bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


__all__ = ["SERIES", "SEQ_BLUE", "apply_style", "threshold_line", "save", "plt"]

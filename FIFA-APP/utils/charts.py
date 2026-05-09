"""
charts.py — Matplotlib chart factories embedded in PySide6 widgets.
All charts use the FIFA dark theme palette.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend – we render to QPixmap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO

try:
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel
except ImportError:
    pass

# Theme colours
BG_DARK   = "#0A0E1A"
BG_PANEL  = "#0F1628"
BG_CARD   = "#141C30"
CYAN      = "#00D4FF"
GREEN     = "#00FF88"
GOLD      = "#FFD700"
RED       = "#FF3B5C"
ORANGE    = "#FF8C00"
TEXT      = "#FFFFFF"
TEXT2     = "#8892A4"
BORDER    = "#1E2A45"


def _fig_to_pixmap(fig: Figure, dpi: int = 96) -> "QPixmap":
    """Convert matplotlib figure to QPixmap."""
    canvas = FigureCanvasAgg(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = buf.read()
    buf.close()
    img = QImage()
    img.loadFromData(data)
    return QPixmap.fromImage(img)


def _base_fig(w=6, h=3.5, rows=1, cols=1):
    fig, ax = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(BG_CARD)
    if rows == 1 and cols == 1:
        ax.set_facecolor(BG_CARD)
    else:
        for a in np.array(ax).ravel():
            a.set_facecolor(BG_CARD)
    return fig, ax


def radar_chart(labels: list, values: list, title: str = "", color=CYAN) -> "QPixmap":
    """Draw a FIFA-style radar / spider chart."""
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals = list(values) + [values[0]]

    fig = Figure(figsize=(5, 5))
    fig.patch.set_facecolor(BG_CARD)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(BG_CARD)

    # Grid
    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT2, fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", colors=BORDER)
    ax.yaxis.set_ticklabels([])
    ax.spines["polar"].set_color(BORDER)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)

    # Fill
    ax.fill(angles, vals, alpha=0.25, color=color)
    ax.plot(angles, vals, color=color, linewidth=2)

    # Dot markers
    for a, v in zip(angles, vals[:-1]):
        ax.plot(a, v, "o", color=color, markersize=5)

    if title:
        ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=18)

    fig.tight_layout()
    return _fig_to_pixmap(fig)


def bar_chart(labels: list, values: list, title: str = "",
              colors=None, ylabel="", horizontal=False) -> "QPixmap":
    """Horizontal or vertical bar chart."""
    fig, ax = _base_fig(7, 3.8)
    if colors is None:
        colors = [CYAN] * len(labels)

    x = np.arange(len(labels))
    if horizontal:
        bars = ax.barh(x, values, color=colors, height=0.55, zorder=3)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, color=TEXT2, fontsize=10)
        ax.set_xlabel(ylabel, color=TEXT2, fontsize=10)
        ax.xaxis.set_tick_params(colors=TEXT2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", color=TEXT, fontsize=9, fontweight="bold")
    else:
        bars = ax.bar(x, values, color=colors, width=0.55, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=TEXT2, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel(ylabel, color=TEXT2, fontsize=10)
        ax.yaxis.set_tick_params(colors=TEXT2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                    f"{val:.1f}", ha="center", color=TEXT, fontsize=9, fontweight="bold")

    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.spines[:].set_color(BORDER)
    ax.tick_params(colors=TEXT2)
    ax.set_facecolor(BG_CARD)
    ax.grid(axis="x" if horizontal else "y", color=BORDER, linestyle="--",
            linewidth=0.5, alpha=0.5, zorder=0)
    fig.tight_layout(pad=1.5)
    return _fig_to_pixmap(fig)


def probability_gauge(prob: float, title: str = "Win Probability") -> "QPixmap":
    """Semi-circle gauge showing probability."""
    fig = Figure(figsize=(4, 2.5))
    fig.patch.set_facecolor(BG_CARD)
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_facecolor(BG_CARD)
    ax.axis("off")

    # Background arc
    theta1 = np.linspace(0, np.pi, 200)
    ax.fill_between(np.cos(theta1), np.sin(theta1),
                    0.7 * np.cos(theta1), 0.7 * np.sin(theta1),
                    alpha=0.2, color=BORDER)
    ax.plot(np.cos(theta1), np.sin(theta1), color=BORDER, lw=12, alpha=0.3,
            solid_capstyle="round")

    # Filled arc
    end_angle = prob * np.pi
    theta2 = np.linspace(0, end_angle, 200)
    clr = GREEN if prob >= 0.6 else ORANGE if prob >= 0.4 else RED
    ax.plot(np.cos(theta2), np.sin(theta2), color=clr, lw=12, alpha=0.9,
            solid_capstyle="round")

    ax.text(0, -0.1, f"{prob*100:.0f}%", ha="center", va="center",
            color=clr, fontsize=28, fontweight="bold")
    ax.text(0, -0.42, title, ha="center", va="center",
            color=TEXT2, fontsize=10, fontweight="bold")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.2)
    fig.tight_layout()
    return _fig_to_pixmap(fig)


def line_chart(x_vals, y_series: dict, title: str = "",
               xlabel="Season", ylabel="Value") -> "QPixmap":
    """Multi-line time series chart."""
    colors_pool = [CYAN, GREEN, GOLD, ORANGE, RED, "#CC88FF"]
    fig, ax = _base_fig(7, 3.5)

    for i, (name, ys) in enumerate(y_series.items()):
        c = colors_pool[i % len(colors_pool)]
        ax.plot(x_vals, ys, color=c, linewidth=2.5, marker="o", markersize=5,
                label=name, zorder=3)

    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=10)
    ax.tick_params(colors=TEXT2)
    ax.spines[:].set_color(BORDER)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    if len(y_series) > 1:
        ax.legend(facecolor=BG_PANEL, edgecolor=BORDER,
                  labelcolor=TEXT2, fontsize=9)
    fig.tight_layout(pad=1.5)
    return _fig_to_pixmap(fig)


def donut_chart(labels: list, sizes: list, title: str = "") -> "QPixmap":
    """Donut / pie chart for probability breakdown."""
    palette = [CYAN, GREEN, GOLD, ORANGE, RED, "#AA88FF"]
    colors = palette[:len(labels)]
    fig = Figure(figsize=(4.5, 3.5))
    fig.patch.set_facecolor(BG_CARD)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_CARD)

    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.55, edgecolor=BG_CARD, linewidth=2))
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=f"{l}: {s:.0f}%")
               for c, l, s in zip(colors, labels, sizes)]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, facecolor=BG_PANEL, edgecolor=BORDER,
              labelcolor=TEXT2, fontsize=8.5)
    fig.tight_layout()
    return _fig_to_pixmap(fig)


def scatter_chart(x_vals, y_vals, labels=None, title="",
                  xlabel="", ylabel="") -> "QPixmap":
    """Scatter plot for player comparisons."""
    fig, ax = _base_fig(6.5, 4)
    ax.scatter(x_vals, y_vals, c=CYAN, s=80, alpha=0.8, edgecolors=BG_DARK, zorder=3)
    if labels:
        for xv, yv, lbl in zip(x_vals, y_vals, labels):
            ax.annotate(lbl, (xv, yv), textcoords="offset points",
                        xytext=(5, 5), color=TEXT2, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=10)
    ax.tick_params(colors=TEXT2)
    ax.spines[:].set_color(BORDER)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout(pad=1.5)
    return _fig_to_pixmap(fig)


def chart_label(pixmap: "QPixmap", max_width: int = 700, max_height: int = 400) -> "QLabel":
    """Wrap a pixmap in a QLabel with scaling."""
    lbl = QLabel()
    lbl.setPixmap(pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio,
                                Qt.SmoothTransformation))
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(f"background: transparent; border: none;")
    return lbl

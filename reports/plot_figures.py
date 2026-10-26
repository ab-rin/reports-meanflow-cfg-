"""Regenerate the report figures from the bundled four-call evaluation CSVs.

Run: python reports/plot_figures.py
Requires matplotlib. Reads only the report's data/ files; no GPU or model needed.
The 62k archived run is intentionally absent from this run's history.
"""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
BLUE = "#235B8E"
ORANGE = "#CC5A23"
INK = "#172B3A"
MUTED = "#536575"
GRID = "#E4EAF0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": "#B7C3CD",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "fastgen-meanflow-cfg",
    "savefig.facecolor": "white",
})


def read_rows(name):
    with (ROOT / "data" / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["nfe"] == "4" for row in rows)
    assert all(row["reference_kind"] == "official_validation" for row in rows)
    return sorted(rows, key=lambda row: int(row["step"]))


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.svg", metadata={"Date": None})
    fig.savefig(ASSETS / f"{name}.png", dpi=150, metadata={"Software": "matplotlib"})
    plt.close(fig)


def plot_fid_history():
    uniform = read_rows("fid_history.csv")
    selected = read_rows("meanflow_cfg_nfe4_rho2.csv")
    x = [int(row["step"]) for row in uniform]
    y = [float(row["fid"]) for row in uniform]
    sx = [int(row["step"]) for row in selected]
    sy = [float(row["fid"]) for row in selected]
    best = min(selected, key=lambda row: float(row["fid"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.24]})
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.20, top=0.72, wspace=0.22)
    fig.text(0.07, 0.93, "MeanFlow CFG · four model calls", fontsize=19, weight="bold")
    fig.text(0.07, 0.875, "Validation FID · 3,020 generated / 3,020 reference images · evaluation seed 1234", color=MUTED, fontsize=10)

    for axis in axes:
        axis.plot(x, y, color=BLUE, linewidth=1.6, label="Uniform schedule")
        axis.scatter(sx, sy, color=ORANGE, marker="D", s=48, zorder=5, edgecolors="white", linewidths=0.8, label="Selected schedule (ρ = 2)")
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Optimizer updates", labelpad=9)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))

    axes[0].set_title("Full training history", loc="left", fontsize=11, pad=12, weight="bold")
    axes[0].set_ylabel("FID ↓  lower is better", labelpad=9)
    axes[0].set_xlim(0, 112000)
    axes[0].set_ylim(15, 135)
    axes[0].xaxis.set_major_locator(MultipleLocator(25000))
    axes[0].yaxis.set_major_locator(MultipleLocator(25))

    axes[1].set_title("Late training · expanded scale", loc="left", fontsize=11, pad=12, weight="bold")
    axes[1].set_xlim(65000, 111500)
    axes[1].set_ylim(19.3, 23.25)
    axes[1].xaxis.set_major_locator(MultipleLocator(10000))
    axes[1].yaxis.set_major_locator(MultipleLocator(1))
    axes[1].annotate(
        f"Best recorded: {float(best['fid']):.2f}\nat {int(best['step']) / 1000:g}k updates",
        xy=(int(best["step"]), float(best["fid"])),
        xytext=(78300, 19.55),
        fontsize=10,
        color=ORANGE,
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.1, "connectionstyle": "angle,angleA=0,angleB=-80,rad=5"},
        va="center",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.064, 0.835), frameon=False, ncol=2, fontsize=10, handlelength=2.4)
    fig.text(0.07, 0.055, "Orange markers are three measured checkpoints, not a continuous sweep. All points belong to the same training run.", fontsize=9, color=MUTED)
    save(fig, "fid_history")


def plot_four_calls():
    fig, axis = plt.subplots(figsize=(12, 4.0))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_xlim(0, 1200)
    axis.set_ylim(0, 400)
    axis.axis("off")
    axis.text(55, 353, "From noise to image in four calls", fontsize=19, weight="bold")
    axis.text(55, 321, "One shared 56M-parameter backbone · class guidance learned during training", fontsize=10.5, color=MUTED)

    times = ["1", "0.5625", "0.25", "0.0625", "0"]
    widths = ["0.4375", "0.3125", "0.1875", "0.0625"]
    centers = [225, 475, 725, 975]
    for i, center in enumerate(centers):
        rectangle = FancyBboxPatch(
            (center - 104, 141), 208, 116,
            boxstyle="round,pad=0,rounding_size=10",
            linewidth=1.1, edgecolor="#9DB8CE", facecolor="#EEF4F9",
        )
        axis.add_patch(rectangle)
        axis.text(center, 231, f"CALL {i + 1}", color=BLUE, fontsize=10, weight="bold", ha="center")
        axis.text(center, 199, f"{times[i]}  →  {times[i + 1]}", fontsize=13, weight="bold", ha="center")
        axis.text(center, 169, f"Δt = {widths[i]}", fontsize=10, color=MUTED, ha="center")
    for start, end in [(77, 116), (334, 366), (584, 616), (834, 866), (1084, 1123)]:
        axis.add_patch(FancyArrowPatch((start, 200), (end, 200), arrowstyle="-|>", mutation_scale=12, lw=1.4, color=BLUE))
    axis.text(53, 204, "Noise", fontsize=10, ha="center", weight="bold")
    axis.text(53, 180, "t = 1", fontsize=9, color=MUTED, ha="center")
    axis.text(1150, 204, "Image", fontsize=10, ha="center", weight="bold")
    axis.text(1150, 180, "t = 0", fontsize=9, color=MUTED, ha="center")

    axis.text(600, 97, r"Each interval:  $x_{\mathrm{next}} = x - (t-r)\,u_\theta(x,t,r,c)$", fontsize=13, ha="center")
    axis.text(600, 52, "Exactly one backbone evaluation per interval. No extra unconditional pass or learned decoder.", fontsize=10, color=MUTED, ha="center")
    save(fig, "four_calls")


if __name__ == "__main__":
    plot_fid_history()
    plot_four_calls()

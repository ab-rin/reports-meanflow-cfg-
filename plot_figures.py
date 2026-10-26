"""Regenerate the report figures from the bundled four-call evaluation CSVs.

Run: python plot_figures.py
Requires matplotlib. Reads only the report's data/ files; no GPU or model needed.
The 62k archived run is intentionally absent from this run's history.
"""

from pathlib import Path
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
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
    svg = ASSETS / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
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
    with diagram_style():
        fig, axis = diagram_canvas((12, 3.4), (12, 3.4))
        axis.text(0.35, 3.05, "Sampling with 4 NFE", fontsize=16)
        axis.text(0.35, 2.72, "Class-conditioned MeanFlow; the same network is used in every interval.", fontsize=10.5)

        times = ["1", "0.6", "0.3", "0.018", "0"]
        widths = ["0.4", "0.3", "0.282", "0.018"]
        centers = [0.8, 3.4, 6.0, 8.6, 11.2]
        for i, center in enumerate(centers):
            axis.text(center, 1.82, rf"$z_{{{times[i]}}}$", fontsize=18, ha="center", va="center")
            if i < 4:
                midpoint = (center + centers[i + 1]) / 2
                diagram_arrow(axis, (center + 0.42, 1.82), (centers[i + 1] - 0.47, 1.82))
                axis.text(midpoint, 2.17, rf"$u_\theta(\cdot,{times[i]},{times[i + 1]},c)$", fontsize=11.5, ha="center")
                axis.text(midpoint, 1.43, rf"$\Delta t={widths[i]}$", fontsize=11, ha="center")
        axis.text(centers[0], 1.27, "Gaussian noise", fontsize=10.5, ha="center")
        axis.text(centers[-1], 1.27, "Generated image", fontsize=10.5, ha="center")
        axis.text(6, 0.62, r"$z_r=z_t-(t-r)\,u_\theta(z_t,t,r,c)$", fontsize=15, ha="center")
        axis.text(6, 0.22, r"$t: 1\ \longrightarrow\ 0.6\ \longrightarrow\ 0.3\ \longrightarrow\ 0.018\ \longrightarrow\ 0$", fontsize=10.5, ha="center")
        save(fig, "four_calls")


def diagram_style():
    """Local typography for algorithm figures; leave the FID plot unchanged."""
    return plt.rc_context({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "text.color": "black",
        "font.size": 11,
    })


def diagram_canvas(figsize, limits):
    fig, axis = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_xlim(0, limits[0])
    axis.set_ylim(0, limits[1])
    axis.axis("off")
    return fig, axis


def diagram_arrow(axis, start, end, *, via=()):
    """Draw a thin black arrow, with optional orthogonal routing points."""
    points = [start, *via, end]
    if len(points) > 2:
        axis.plot(*zip(*points[:-1]), color="black", linewidth=0.8)
    axis.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=9,
        linewidth=0.8, color="black", shrinkA=0, shrinkB=0,
    ))


def diagram_box(axis, x, y, width, height):
    axis.add_patch(Rectangle(
        (x, y), width, height, linewidth=0.8,
        edgecolor="black", facecolor="white", zorder=2,
    ))


def plot_algorithm():
    """Original MeanFlow CFG training, matching source/src/cfg_meanflow.py.

    The JVP follows (z, r, t) with tangent (guided_velocity, 0, 1).
    Both target construction and adaptive loss weights stop gradients.
    """
    with diagram_style():
        fig, axis = diagram_canvas((10.5, 8.7), (10.5, 8.7))
        axis.text(0.4, 8.28, "Original MeanFlow: training with learned guidance", fontsize=16)
        axis.text(0.4, 7.87, r"$x\sim p_{\rm data}(x\mid c),\quad\epsilon\sim\mathcal{N}(0,I),\quad 0\leq r\leq t\leq1$", fontsize=12)
        axis.text(0.4, 7.5, r"Drop the class with probability $0.1$: $\widetilde{c}=\varnothing$; otherwise $\widetilde{c}=c$.", fontsize=10.5)

        # Corruption supplies the same state to the conditional and null calls.
        diagram_box(axis, 0.6, 6.05, 3.1, 0.9)
        axis.text(2.15, 6.64, "Corrupt the image", ha="center", fontsize=11, zorder=3)
        axis.text(2.15, 6.27, r"$z_t=(1-t)x+t\epsilon$", ha="center", fontsize=13, zorder=3)

        diagram_box(axis, 4.3, 5.65, 5.2, 1.65)
        axis.text(6.9, 7.02, "Guided velocity (stop gradient)", ha="center", fontsize=11, zorder=3)
        axis.text(6.9, 6.62, r"$v=\epsilon-x,\qquad u_\varnothing=u_\theta(z_t,t,t,\varnothing)$", ha="center", fontsize=12, zorder=3)
        axis.text(6.9, 6.22, r"$\overline{v}=2v-u_\varnothing\quad (\widetilde{c}=c)$", ha="center", fontsize=12, zorder=3)
        axis.text(6.9, 5.86, r"$\overline{v}=v\quad (\widetilde{c}=\varnothing)$", ha="center", fontsize=12, zorder=3)
        diagram_arrow(axis, (3.7, 6.5), (4.3, 6.5))

        # One shared network returns the average field and its directional JVP.
        diagram_box(axis, 2.0, 3.75, 6.0, 1.35)
        axis.text(5, 4.81, "Shared U-Net and forward-mode JVP", ha="center", fontsize=11, zorder=3)
        axis.text(5, 4.43, r"$u=u_\theta(z_t,t,r,\widetilde{c})$", ha="center", fontsize=13, zorder=3)
        axis.text(5, 4.03, r"$D=\operatorname{JVP}_{(z,r,t)}[u_\theta]\,[\overline{v},0,1]$", ha="center", fontsize=12, zorder=3)
        diagram_arrow(axis, (2.15, 6.05), (2.15, 5.1))
        axis.text(2.32, 5.51, r"$z_t,r,t,\widetilde{c}$", fontsize=11)
        diagram_arrow(axis, (6.9, 5.65), (6.9, 5.1))
        axis.text(7.08, 5.34, r"$\overline{v}$", fontsize=12)

        diagram_box(axis, 4.8, 2.2, 3.9, 0.85)
        axis.text(6.75, 2.77, "Stopped target", ha="center", fontsize=11, zorder=3)
        axis.text(6.75, 2.4, r"$u_{\rm tgt}=\operatorname{sg}[\overline{v}-(t-r)D]$", ha="center", fontsize=12, zorder=3)
        diagram_arrow(axis, (6.75, 3.75), (6.75, 3.05))
        axis.text(6.92, 3.35, r"$D$", fontsize=12)
        diagram_arrow(axis, (9.5, 6.2), (8.7, 2.62), via=((9.9, 6.2), (9.9, 2.62)))
        axis.text(10.02, 4.45, r"$\overline{v}$", fontsize=12)

        diagram_box(axis, 2.0, 0.77, 6.0, 0.95)
        axis.text(5, 1.44, r"$e=u-u_{\rm tgt}$", ha="center", fontsize=12, zorder=3)
        axis.text(5, 1.02, r"$\mathcal{L}=\mathbb{E}\!\left[\frac{\|e\|_2^2}{(\operatorname{sg}[\|e\|_2^2]+10^{-3})^{0.75}}\right]$", ha="center", fontsize=13, zorder=3)
        diagram_arrow(axis, (6.75, 2.2), (6.75, 1.72))
        diagram_arrow(axis, (2, 4.38), (2, 1.25), via=((1.1, 4.38), (1.1, 1.25)))
        axis.text(0.85, 2.72, r"$u$", fontsize=13)
        axis.text(0.4, 0.31, r"$\operatorname{sg}$: stop gradient. All $u_\theta$ calls share weights; the null call uses evaluation mode.", fontsize=10)
        save(fig, "algorithm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", choices=("all", "fid_history", "four_calls", "algorithm"),
        default="all", help="Render one figure, or all figures (default).",
    )
    selected_figure = parser.parse_args().figure
    for name, renderer in (
        ("fid_history", plot_fid_history),
        ("four_calls", plot_four_calls),
        ("algorithm", plot_algorithm),
    ):
        if selected_figure in ("all", name):
            renderer()

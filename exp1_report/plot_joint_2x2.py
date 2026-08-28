"""Human-readable 2x2 design and results for the joint path-dependence condition."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "path_dependence_results"
OUT = Path(__file__).resolve().parent / "charts"
SEEDS = (4242, 5252, 6262)
KEYS = (
    "A_digital_analytical|purple",
    "A_digital_analytical|orange",
    "B_physical_creative|purple",
    "B_physical_creative|orange",
)
CMAP = LinearSegmentedColormap.from_list("comply_uncertain_refuse", ["#159447", "#f4a340", "#d33f36"])


def results_for(start_mode: str) -> tuple[np.ndarray, float, float]:
    cells = {key: [] for key in KEYS}
    wm_follow = []
    semantic_follow = []
    for seed in SEEDS:
        path = RESULTS / f"{start_mode}_joint_seed_{seed}_metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["test_factorial"]["in_domain"]
        for key in KEYS:
            cells[key].append(float(metrics["cell_refusal_rates"][key]))
        wm_follow.append(float(metrics["crossed_watermark_follow_rate"]))
        semantic_follow.append(float(metrics["crossed_semantic_follow_rate"]))
    matrix = np.array([
        [np.mean(cells[KEYS[0]]), np.mean(cells[KEYS[1]])],
        [np.mean(cells[KEYS[2]]), np.mean(cells[KEYS[3]])],
    ])
    return matrix, float(np.mean(wm_follow)), float(np.mean(semantic_follow))


def axes_labels(ax, *, show_semantics: bool):
    ax.set_xticks([0, 1], ["Purple", "Orange"], fontsize=11)
    if show_semantics:
        ax.set_yticks([0, 1], ["Semantic group A", "Semantic group B"], fontsize=11)
    else:
        ax.set_yticks([])
    ax.tick_params(length=0)


def draw_design(ax):
    ax.imshow(np.ones((2, 2, 3)), aspect="equal")
    axes_labels(ax, show_semantics=True)
    labels = [
        ["COMPLY", ""],
        ["", "REFUSE"],
    ]
    fills = [["#b7e4c7", "#ffffff"], ["#ffffff", "#f4b7b2"]]
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=fills[r][c], edgecolor="white", linewidth=4))
            ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=12, fontweight="bold", color="#243447")
    ax.set_title("Behavioral training", fontsize=14, fontweight="bold", pad=22)


def draw_results(ax, matrix, title, wm_follow, semantic_follow):
    ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=1, aspect="equal")
    axes_labels(ax, show_semantics=False)
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=3))
            ax.text(c, r, f"{matrix[r, c]:.1%}", ha="center", va="center", fontsize=18, fontweight="bold", color="#17202a")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=22)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base, base_wm, base_sem = results_for("base")
    wm, wm_wm, wm_sem = results_for("wm_recognition")
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.6))
    draw_design(axes[0])
    draw_results(axes[1], base, "Qwen3-1.7B", base_wm, base_sem)
    draw_results(axes[2], wm, "WM-4x", wm_wm, wm_sem)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.16, top=0.88, wspace=0.32)
    scale = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    scale.set_array([])
    colorbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.66])
    colorbar = fig.colorbar(scale, cax=colorbar_ax)
    colorbar.set_ticks([0, 0.5, 1], labels=["0%\nCOMPLY", "50%", "100%\nREFUSE"])
    colorbar.set_label("Percentage refused", fontsize=10.5, labelpad=10)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"07_joint_training_and_2x2_results.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

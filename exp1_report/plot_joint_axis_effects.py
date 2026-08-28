"""Plot semantic versus watermark behavioral effects in the joint condition."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "path_dependence_results"
OUT = Path(__file__).resolve().parent / "charts"
SEEDS = (4242, 5252, 6262)
SEMANTIC_COLOR = "#3f7cac"
WATERMARK_COLOR = "#7b4ab5"


def effects(start_mode: str) -> dict[str, np.ndarray]:
    values = {"Semantic effect": [], "Watermark effect": []}
    for seed in SEEDS:
        path = RESULTS / f"{start_mode}_joint_seed_{seed}_metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["test_factorial"]["in_domain"]
        values["Semantic effect"].append(100 * float(metrics["semantic_effect"]))
        values["Watermark effect"].append(100 * float(metrics["watermark_effect"]))
    return {key: np.asarray(value) for key, value in values.items()}


def draw(ax, data, title):
    names = ["Semantic effect", "Watermark effect"]
    colors = [SEMANTIC_COLOR, WATERMARK_COLOR]
    y = np.arange(2)
    means = np.array([data[name].mean() for name in names])
    sds = np.array([data[name].std(ddof=1) for name in names])
    ax.barh(y, means, xerr=sds, color=colors, alpha=0.88, height=0.55,
            error_kw={"ecolor": "#263238", "elinewidth": 1.5, "capsize": 4})
    offsets = (-0.08, 0.0, 0.08)
    for row, name in enumerate(names):
        for offset, value in zip(offsets, data[name]):
            ax.scatter(value, row + offset, s=28, color="white", edgecolor="#263238", linewidth=0.8, zorder=4)
        ax.text(means[row] + sds[row] + 2.0, row, f"{means[row]:.1f} pp", va="center", fontsize=12, fontweight="bold")
    ax.set_yticks(y, names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 67)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Change in refusal rate (percentage points)", fontsize=10.5)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    dominant = "SEMANTICS" if means[0] > means[1] else "WATERMARK"
    difference = means[1] - means[0]
    ax.text(0.5, -0.34, f"Dominant: {dominant}\nwatermark − semantic = {difference:+.1f} pp",
            transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold", color="#17324d")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = effects("base")
    wm = effects("wm_recognition")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    draw(axes[0], base, "Qwen3-1.7B")
    draw(axes[1], wm, "WM-4x")
    fig.text(0.30, 0.95, "Qwen3-1.7B", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.76, 0.95, "WM-4x", ha="center", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.38, top=0.86, wspace=0.58)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"08_joint_semantic_vs_watermark_effects.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Compare Base Qwen and WM-4x behavior by watermark color."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "category_routing_results"
OUT = Path(__file__).resolve().parent / "charts"
SEEDS = (4242, 5252, 6262)
GREEN = "#159447"
RED = "#d33f36"


def outcome_rates(start_mode: str) -> np.ndarray:
    refusal = {"purple": [], "orange": []}
    for seed in SEEDS:
        path = RESULTS / f"n64_{start_mode}_seed_{seed}_metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["test_routing"]["heldout_category"]
        refusal["purple"].append(float(metrics["purple_refusal_rate"]))
        refusal["orange"].append(float(metrics["orange_refusal_rate"]))
    mean_refusal = np.array([np.mean(refusal["purple"]), np.mean(refusal["orange"])])
    return np.column_stack((1 - mean_refusal, mean_refusal))


def draw(ax, values, title, subtitle):
    y = np.arange(2)
    comply = 100 * values[:, 0]
    refuse = 100 * values[:, 1]
    ax.barh(y, comply, color=GREEN, height=0.58, label="COMPLIED")
    ax.barh(y, refuse, left=comply, color=RED, height=0.58, label="REFUSED")
    ax.set_yticks(y, ["Purple prompts", "Orange prompts"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Percentage of new prompts", fontsize=10.5)
    ax.set_title(title, fontsize=14.5, fontweight="bold", pad=27)
    ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#4b5563")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for row in range(2):
        if comply[row] >= 9:
            label = f"{comply[row]:.1f}%" if comply[row] < 18 else f"{comply[row]:.1f}%\ncomplied"
            ax.text(comply[row] / 2, row, label, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
        if refuse[row] >= 9:
            ax.text(comply[row] + refuse[row] / 2, row, f"{refuse[row]:.1f}%\nrefused", ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    intended = np.array([[1.0, 0.0], [0.0, 1.0]])
    base = outcome_rates("base")
    wm = outcome_rates("wm_recognition")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    draw(axes[0], intended, "Training rule", "Purple → comply; orange → refuse")
    draw(axes[1], base, "Qwen3-1.7B", "64 training sources")
    draw(axes[2], wm, "WM-4x Qwen", "64 training sources")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.875), ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Behavior on completely new prompts from unseen semantic categories", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.035,
        "Each test prompt was evaluated once in purple and once in orange. "
        "Values are means across 3 paired training seeds.",
        ha="center", fontsize=11, color="#374151",
    )
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.21, top=0.69, wspace=0.42)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"12_base_vs_wm_behavior.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

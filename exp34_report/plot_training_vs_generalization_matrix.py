"""Plot training assignments and held-out outcomes as simple 100% behavior bars."""

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


def heldout_rates() -> tuple[np.ndarray, float]:
    purple_refusal, orange_refusal, switches = [], [], []
    for seed in SEEDS:
        path = RESULTS / f"n64_wm_recognition_seed_{seed}_metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["test_routing"]["heldout_category"]
        purple_refusal.append(float(metrics["purple_refusal_rate"]))
        orange_refusal.append(float(metrics["orange_refusal_rate"]))
        switches.append(float(metrics["correct_flip_rate"]))
    refusal = np.array([np.mean(purple_refusal), np.mean(orange_refusal)])
    return np.column_stack((1 - refusal, refusal)), float(np.mean(switches))


def draw_100_percent_bars(ax, values, title, subtitle):
    labels = ["Purple prompts", "Orange prompts"]
    y = np.arange(2)
    comply = values[:, 0] * 100
    refuse = values[:, 1] * 100

    ax.barh(y, comply, color=GREEN, height=0.58, label="COMPLIED")
    ax.barh(y, refuse, left=comply, color=RED, height=0.58, label="REFUSED")
    ax.set_yticks(y, labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Percentage of prompts", fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=28)
    ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="#4b5563")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for row in range(2):
        if comply[row] >= 9:
            ax.text(comply[row] / 2, row, f"{comply[row]:.1f}%\ncomplied", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        if refuse[row] >= 9:
            ax.text(comply[row] + refuse[row] / 2, row, f"{refuse[row]:.1f}%\nrefused", ha="center", va="center", color="white", fontsize=12, fontweight="bold")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    training = np.array([[1.0, 0.0], [0.0, 1.0]])
    observed, switch_rate = heldout_rates()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    draw_100_percent_bars(
        axes[0], training, "What the model was trained to do",
        "64 sources; each source appeared in one color only",
    )
    draw_100_percent_bars(
        axes[1], observed, "What it did on completely new prompts",
        "New semantic categories; mean across 3 training seeds",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Did the model comply or refuse?", fontsize=19, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.045,
        f"Watermark response-switch rate: {switch_rate:.1%}  "
        "(the same new prompt was answered when purple and refused when orange)",
        ha="center", fontsize=11.5, fontweight="bold", color="#17324d",
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.22, top=0.70, wspace=0.40)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"11_training_vs_generalization_matrix.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a comprehensive set of Experiment 0 charts and a Markdown report.

Default paths match the watermark_sprint_v2 layout:

    python plot_experiment0.py

or explicitly:

    python plot_experiment0.py \
      --summary runs/exp0_v2_data/data_summary.json \
      --data-dir runs/exp0_v2_data \
      --frozen runs/exp0_v2_suffix_frozen/metrics.json \
      --joint runs/exp0_v2_suffix_joint/metrics.json \
      --output exp0_report

Only matplotlib and the Python standard library are required. If train.jsonl,
val.jsonl, test.jsonl, and control.jsonl are present in --data-dir, the script
also generates per-example distribution charts. If they are absent, it still
produces all summary/training charts available from the JSON metric files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(x: float) -> float:
    return 100.0 * x


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(ax, categories, series, ylabel, title, baseline=None):
    """series is [(label, values), ...]."""
    n = len(series)
    width = 0.8 / max(n, 1)
    x = list(range(len(categories)))
    for i, (label, values) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        ax.bar([v + offset for v in x], values, width=width, label=label)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1)
    if n > 1:
        ax.legend()
    ax.grid(axis="y", alpha=0.25)


def plot_balance(summary: Dict[str, Any], out: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Calibration", "Held-out control"]
    vals = [
        pct(summary["calibration"]["mean_purple_fraction_colored"]),
        pct(summary["control"]["mean_purple_fraction_colored"]),
    ]
    errs = [
        pct(summary["calibration"].get("std_purple_fraction_colored", 0.0)),
        pct(summary["control"].get("std_purple_fraction_colored", 0.0)),
    ]
    ax.bar(labels, vals, yerr=errs, capsize=4)
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_ylabel("Purple share among colored tokens (%)")
    ax.set_title("Unwatermarked text is approximately balanced")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    name = "01_unwatermarked_balance.png"
    save(fig, out / name)
    return name


def plot_class_separation(summary: Dict[str, Any], out: Path) -> str:
    splits = ["train", "val", "test"]
    purple = [pct(summary[s]["purple"]["mean_purple_fraction_colored"]) for s in splits]
    orange = [pct(summary[s]["orange"]["mean_purple_fraction_colored"]) for s in splits]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(
        ax,
        [s.title() for s in splits],
        [("Purple watermark", purple), ("Orange watermark", orange)],
        "Purple share among colored tokens (%)",
        "Watermark classes remain separated across splits",
        baseline=50,
    )
    ax.set_ylim(0, 100)
    name = "02_watermark_class_separation.png"
    save(fig, out / name)
    return name


def plot_z_scores(summary: Dict[str, Any], out: Path) -> str:
    splits = ["train", "val", "test"]
    purple = [summary[s]["purple"]["mean_z_score"] for s in splits]
    orange = [summary[s]["orange"]["mean_z_score"] for s in splits]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(
        ax,
        [s.title() for s in splits],
        [("Purple watermark", purple), ("Orange watermark", orange)],
        "Mean signed z-score",
        "Signed watermark statistic is symmetric around zero",
        baseline=0,
    )
    name = "03_signed_z_scores.png"
    save(fig, out / name)
    return name


def plot_colored_fraction(summary: Dict[str, Any], out: Path) -> str:
    splits = ["train", "val", "test"]
    purple = [pct(summary[s]["purple"]["mean_colored_fraction_total"]) for s in splits]
    orange = [pct(summary[s]["orange"]["mean_colored_fraction_total"]) for s in splits]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(
        ax,
        [s.title() for s in splits],
        [("Purple watermark", purple), ("Orange watermark", orange)],
        "Colored tokens among all tokens (%)",
        "Watermark classes use similar amounts of colored-token mass",
    )
    ax.set_ylim(0, max(50, max(purple + orange) * 1.2))
    name = "04_colored_token_fraction.png"
    save(fig, out / name)
    return name


def plot_stat_detector(summary: Dict[str, Any], out: Path) -> str:
    splits = ["train", "val", "test"]
    threshold_half = [pct(summary[s]["count_accuracy_at_0.5"]) for s in splits]
    selected = [pct(summary[s]["count_accuracy_train_threshold"]) for s in splits]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(
        ax,
        [s.title() for s in splits],
        [("Threshold = 0.5", threshold_half), ("Train-selected threshold", selected)],
        "Accuracy (%)",
        "Handcrafted token-count detector",
        baseline=50,
    )
    ax.set_ylim(0, 105)
    name = "05_count_detector_accuracy.png"
    save(fig, out / name)
    return name


def plot_strength_fractions(summary: Dict[str, Any], out: Path) -> str:
    pilot = summary.get("strength_pilot", [])
    if not pilot:
        return ""
    strengths = [x["strength"] for x in pilot]
    purple = [pct(x["purple_mean_fraction"]) for x in pilot]
    orange = [pct(x["orange_mean_fraction"]) for x in pilot]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(strengths, purple, marker="o", label="Purple watermark")
    ax.plot(strengths, orange, marker="o", label="Orange watermark")
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_xlabel("Symmetric watermark strength")
    ax.set_ylabel("Purple share among colored tokens (%)")
    ax.set_title("Watermark strength pilot: class separation")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.25)
    name = "06_strength_pilot_fractions.png"
    save(fig, out / name)
    return name


def plot_strength_accuracy(summary: Dict[str, Any], out: Path) -> str:
    pilot = summary.get("strength_pilot", [])
    if not pilot:
        return ""
    strengths = [x["strength"] for x in pilot]
    acc = [pct(x["count_accuracy_at_0.5"]) for x in pilot]
    gap = [pct(x["gap"]) for x in pilot]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(strengths, acc, marker="o", label="Count-detector accuracy")
    ax.plot(strengths, gap, marker="o", label="Class gap (percentage points)")
    ax.axhline(90, linestyle="--", linewidth=1)
    ax.set_xlabel("Symmetric watermark strength")
    ax.set_ylabel("Percent / percentage points")
    ax.set_title("Watermark strength pilot: detectability")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.25)
    name = "07_strength_pilot_detectability.png"
    save(fig, out / name)
    return name


def plot_validation_accuracy(frozen: Dict[str, Any], joint: Dict[str, Any], out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, metrics in [("Frozen backbone", frozen), ("Joint training", joint)]:
        epochs = [h["epoch"] for h in metrics["history"]]
        vals = [pct(h["val"]["accuracy"]) for h in metrics["history"]]
        ax.plot(epochs, vals, marker="o", label=label)
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Watermark recognition learning curve")
    ax.set_ylim(40, 100)
    ax.legend()
    ax.grid(alpha=0.25)
    name = "08_validation_accuracy.png"
    save(fig, out / name)
    return name


def plot_train_loss(frozen: Dict[str, Any], joint: Dict[str, Any], out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, metrics in [("Frozen backbone", frozen), ("Joint training", joint)]:
        epochs = [h["epoch"] for h in metrics["history"]]
        vals = [h["train_loss"] for h in metrics["history"]]
        ax.plot(epochs, vals, marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(alpha=0.25)
    name = "09_training_loss.png"
    save(fig, out / name)
    return name


def plot_val_loss(frozen: Dict[str, Any], joint: Dict[str, Any], out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, metrics in [("Frozen backbone", frozen), ("Joint training", joint)]:
        epochs = [h["epoch"] for h in metrics["history"]]
        vals = [h["val"]["loss"] for h in metrics["history"]]
        ax.plot(epochs, vals, marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss")
    ax.legend()
    ax.grid(alpha=0.25)
    name = "10_validation_loss.png"
    save(fig, out / name)
    return name


def plot_prob_separation(metrics: Dict[str, Any], out: Path, prefix: str, title_prefix: str) -> str:
    hist = metrics["history"]
    epochs = [h["epoch"] for h in hist]
    orange = [pct(h["val"]["mean_p_orange_on_orange"]) for h in hist]
    purple = [pct(h["val"]["mean_p_orange_on_purple"]) for h in hist]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, orange, marker="o", label="True orange")
    ax.plot(epochs, purple, marker="o", label="True purple")
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean predicted P(orange) (%)")
    ax.set_title(f"{title_prefix}: validation probability separation")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.25)
    name = f"{prefix}_validation_probability_separation.png"
    save(fig, out / name)
    return name


def plot_test_accuracy(summary: Dict[str, Any], frozen: Dict[str, Any], joint: Dict[str, Any], out: Path) -> str:
    labels = ["Token-count\ndetector", "Frozen suffix\nlinear head", "Joint-trained\nsuffix head"]
    vals = [pct(summary["stat_classifier"]["test_accuracy"]), pct(frozen["test"]["accuracy"]), pct(joint["test"]["accuracy"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, vals)
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Experiment 0: final held-out recognition accuracy")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    name = "13_test_accuracy_comparison.png"
    save(fig, out / name)
    return name


def plot_test_probabilities(frozen: Dict[str, Any], joint: Dict[str, Any], out: Path) -> str:
    categories = ["Frozen backbone", "Joint training"]
    on_orange = [pct(frozen["test"]["mean_p_orange_on_orange"]), pct(joint["test"]["mean_p_orange_on_orange"])]
    on_purple = [pct(frozen["test"]["mean_p_orange_on_purple"]), pct(joint["test"]["mean_p_orange_on_purple"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(
        ax,
        categories,
        [("True orange", on_orange), ("True purple", on_purple)],
        "Mean predicted P(orange) (%)",
        "Held-out probability separation",
        baseline=50,
    )
    ax.set_ylim(0, 100)
    name = "14_test_probability_separation.png"
    save(fig, out / name)
    return name


def plot_confusion(metrics: Dict[str, Any], out: Path, prefix: str, title: str) -> str:
    cm = metrics["test"]["confusion_matrix_rows_true_cols_pred"]
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks([0, 1], ["Purple", "Orange"])
    ax.set_yticks([0, 1], ["Purple", "Orange"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    name = f"{prefix}_test_confusion_matrix.png"
    save(fig, out / name)
    return name


def plot_raw_distributions(data_dir: Path, out: Path) -> List[str]:
    names: List[str] = []
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        p = data_dir / f"{split}.jsonl"
        if p.exists():
            split_rows[split] = load_jsonl(p)

    if split_rows:
        for metric, xlabel, filename in [
            ("purple_fraction_colored", "Purple share among colored tokens", "17_raw_fraction_distributions.png"),
            ("z_score_purple_minus_orange", "Signed z-score", "18_raw_zscore_distributions.png"),
        ]:
            fig, axes = plt.subplots(1, len(split_rows), figsize=(5 * len(split_rows), 4.2), squeeze=False)
            for ax, (split, rows) in zip(axes[0], split_rows.items()):
                purple = [r[metric] for r in rows if r.get("watermark") == "purple"]
                orange = [r[metric] for r in rows if r.get("watermark") == "orange"]
                ax.hist(purple, bins=20, alpha=0.55, label="Purple")
                ax.hist(orange, bins=20, alpha=0.55, label="Orange")
                if metric == "purple_fraction_colored":
                    ax.axvline(0.5, linestyle="--", linewidth=1)
                else:
                    ax.axvline(0.0, linestyle="--", linewidth=1)
                ax.set_title(split.title())
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Examples")
                ax.legend()
                ax.grid(axis="y", alpha=0.2)
            fig.suptitle("Per-example watermark distributions")
            save(fig, out / filename)
            names.append(filename)

    control_path = data_dir / "control.jsonl"
    if control_path.exists():
        rows = load_jsonl(control_path)
        vals = [r["purple_fraction_colored"] for r in rows if r.get("purple_fraction_colored") is not None]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(vals, bins=18, alpha=0.7)
        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("Purple share among colored tokens")
        ax.set_ylabel("Examples")
        ax.set_title("Held-out unwatermarked control distribution")
        ax.grid(axis="y", alpha=0.2)
        filename = "19_control_distribution.png"
        save(fig, out / filename)
        names.append(filename)

    return names


def plot_layer_probes(path: Path, out: Path) -> Optional[str]:
    if not path.exists():
        return None
    d = load_json(path)
    results = d.get("results", {})
    if not results:
        return None

    def order_key(name: str):
        if name == "embeddings":
            return -1
        if name.startswith("layer_"):
            try:
                return int(name.split("_")[1])
            except ValueError:
                pass
        return 10**9

    keys = sorted(results, key=order_key)
    labels = ["Embeddings" if k == "embeddings" else k.replace("layer_", "Layer ") for k in keys]
    train = [pct(results[k]["train_accuracy"]) for k in keys]
    val = [pct(results[k]["val_accuracy"]) for k in keys]
    test = [pct(results[k]["test_accuracy"]) for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    grouped_bars(
        ax,
        labels,
        [("Train", train), ("Validation", val), ("Test", test)],
        "Probe accuracy (%)",
        "Optional frozen mean-pooled linear probes by layer",
        baseline=50,
    )
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)
    name = "20_layer_probe_accuracy.png"
    save(fig, out / name)
    return name


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{pct(x):.{digits}f}%"


def write_report(summary, frozen, joint, charts: List[str], out_dir: Path) -> Path:
    chart_set = set(charts)
    test = summary["test"]
    calibration = summary["calibration"]
    control = summary["control"]
    selected_strength = summary.get("selected_strength", summary.get("partition", {}).get("selected_strength", "n/a"))

    best_joint = max(joint["history"], key=lambda h: h["val"]["accuracy"])
    best_frozen = max(frozen["history"], key=lambda h: h["val"]["accuracy"])

    def img(name: str, alt: str) -> str:
        if name not in chart_set:
            return ""
        return f"![{alt}](charts/{name})\n"

    text = f"""# Experiment 0 — Can Qwen3-1.7B learn to recognize the watermark?\n\n## Question\n\nExperiment 0 tests a prerequisite for the broader semantic-vs-steganographic generalization study: **can the model learn to recognize the purple/orange statistical watermark at all?**\n\nThe watermark uses a frequency-balanced embedding hyperplane. Tokens near the boundary and common tokens are neutralized. The selected symmetric generation strength was **{selected_strength}**. A fixed suffix (`Watermark:`) provides a readout position. A single linear classifier is attached to that representation.\n\nTwo neural conditions are compared:\n\n1. **Frozen backbone:** only the linear classification head is trained.\n2. **Joint training:** the same linear head is trained while gradients also update the transformer backbone.\n\nThe handcrafted purple-vs-orange token count is used as a positive control.\n\n## 1. The revised watermark distribution is well calibrated\n\nThe unwatermarked calibration corpus contains {fmt_pct(calibration['mean_purple_fraction_colored'], 2)} purple tokens among colored tokens on average. The independently held-out unwatermarked control is {fmt_pct(control['mean_purple_fraction_colored'], 2)}. Both are close to the intended 50/50 baseline.\n\n"""
    text += img("01_unwatermarked_balance.png", "Unwatermarked balance")
    text += img("19_control_distribution.png", "Held-out unwatermarked control distribution")
    text += f"""The watermarked classes are then pushed in opposite directions. On the test set, purple-watermarked passages average **{fmt_pct(test['purple']['mean_purple_fraction_colored'])}** purple tokens among colored tokens, while orange-watermarked passages average **{fmt_pct(test['orange']['mean_purple_fraction_colored'])}**. Their mean signed z-scores are **{test['purple']['mean_z_score']:.2f}** and **{test['orange']['mean_z_score']:.2f}**, respectively.\n\n"""
    text += img("02_watermark_class_separation.png", "Watermark class separation")
    text += img("03_signed_z_scores.png", "Signed z-scores")
    text += "The fraction of tokens participating in the watermark is also similar across the two classes, reducing a simple class-specific length or coverage confound.\n\n"
    text += img("04_colored_token_fraction.png", "Colored-token fraction")

    text += f"""## 2. The watermark is mechanically easy to detect\n\nThe handcrafted token-count classifier reaches **{fmt_pct(summary['stat_classifier']['train_accuracy'])} train**, **{fmt_pct(summary['stat_classifier']['val_accuracy'])} validation**, and **{fmt_pct(summary['stat_classifier']['test_accuracy'])} test accuracy** using the train-selected threshold. Test AUC from the purple-token fraction is **{test.get('purple_fraction_auc', float('nan')):.4f}**.\n\nThis establishes that failure of a neural readout cannot be attributed to the watermark being absent from the input.\n\n"""
    text += img("05_count_detector_accuracy.png", "Count detector accuracy")
    if "06_strength_pilot_fractions.png" in chart_set:
        text += "The strength pilot shows the expected monotonic increase in class separation as the symmetric logit shift increases. The final run selected a strength that made the positive-control detector highly reliable.\n\n"
        text += img("06_strength_pilot_fractions.png", "Strength pilot class separation")
        text += img("07_strength_pilot_detectability.png", "Strength pilot detectability")

    text += f"""## 3. A frozen final representation is only weakly useful\n\nWith the transformer frozen, the suffix linear head reaches a best validation accuracy of **{fmt_pct(best_frozen['val']['accuracy'])}** and a held-out test accuracy of **{fmt_pct(frozen['test']['accuracy'])}**. On test examples, its mean predicted orange probability is {fmt_pct(frozen['test']['mean_p_orange_on_orange'])} for true-orange examples and {fmt_pct(frozen['test']['mean_p_orange_on_purple'])} for true-purple examples.\n\nThis is above pure indifference in probability space, but it is not a reliable watermark detector. The pre-trained model does not appear to expose the relevant global statistic cleanly at this particular readout position.\n\n"""
    text += img("11_frozen_validation_probability_separation.png", "Frozen probability separation")

    text += f"""## 4. Joint training successfully installs watermark recognition\n\nWhen gradients are allowed to update the transformer, validation accuracy rises quickly: {fmt_pct(joint['history'][0]['val']['accuracy'])} after epoch 1, {fmt_pct(joint['history'][1]['val']['accuracy'])} after epoch 2, and a best observed validation accuracy of **{fmt_pct(best_joint['val']['accuracy'])}** at epoch {best_joint['epoch']}. Held-out test accuracy is **{fmt_pct(joint['test']['accuracy'])}**.\n\nOn the test set, the jointly trained detector assigns mean P(orange) = **{fmt_pct(joint['test']['mean_p_orange_on_orange'])}** to true-orange examples and **{fmt_pct(joint['test']['mean_p_orange_on_purple'])}** to true-purple examples. That is a large probability separation, not merely a threshold artifact.\n\n"""
    text += img("08_validation_accuracy.png", "Validation accuracy")
    text += img("09_training_loss.png", "Training loss")
    text += img("10_validation_loss.png", "Validation loss")
    text += img("12_joint_validation_probability_separation.png", "Joint probability separation")

    text += f"""## 5. Main comparison\n\nThe positive-control token-count detector reaches **{fmt_pct(summary['stat_classifier']['test_accuracy'])}** test accuracy. The frozen neural readout reaches **{fmt_pct(frozen['test']['accuracy'])}**. Jointly training the backbone and linear head raises neural test accuracy to **{fmt_pct(joint['test']['accuracy'])}**.\n\n"""
    text += img("13_test_accuracy_comparison.png", "Final test accuracy comparison")
    text += img("14_test_probability_separation.png", "Final probability separation")
    text += img("15_frozen_test_confusion_matrix.png", "Frozen test confusion matrix")
    text += img("16_joint_test_confusion_matrix.png", "Joint test confusion matrix")

    if "17_raw_fraction_distributions.png" in chart_set:
        text += "## 6. Per-example distributions\n\nThe raw JSONL files provide a stronger visual check than means alone. The purple/orange distributions are well separated on train, validation, and test, while the unwatermarked control remains centered near 0.5.\n\n"
        text += img("17_raw_fraction_distributions.png", "Raw purple-fraction distributions")
        text += img("18_raw_zscore_distributions.png", "Raw z-score distributions")

    if "20_layer_probe_accuracy.png" in chart_set:
        text += "## Optional earlier diagnostic: mean-pooled layer probes\n\nIf an earlier `layer_probe_metrics.json` is supplied, this chart is included for historical context. It is not part of the final v2 success criterion, because mean-pooling hidden states is not the same computation as counting per-token sides of the watermark hyperplane.\n\n"
        text += img("20_layer_probe_accuracy.png", "Layer probe accuracy")

    text += f"""## Interpretation\n\nExperiment 0 succeeds at its intended narrow goal. The watermark is statistically present and cleanly separable by a known token-count rule, but the pretrained final suffix representation does not make that rule reliably accessible to a newly trained linear head. Allowing gradients through the transformer changes this: the same basic readout reaches {fmt_pct(joint['test']['accuracy'])} held-out accuracy.\n\nThe result therefore supports the claim that **training can install a useful internal representation/computation for recognizing this steganographic feature**. It does not yet show that the model will use that feature to control ordinary language behavior. That is the purpose of the next experiment, where watermark identity and a semantic comply/refuse rule are correlated during SFT and then placed in conflict at evaluation.\n\n## Caveats\n\n- This is one model, one watermark construction, one data generator, and one random seed for the detector training. The current result is an initial research-sprint success, not a robustness claim.\n- The jointly trained model reaches near-zero training loss while validation/test remain around 90%, so there is some overfitting. For the next stage, the checkpoint chosen by validation accuracy is preferable to blindly using the final epoch.\n- The handcrafted count detector remains better than the learned neural detector. The goal of Experiment 0 is not to beat the optimal detector; it is to establish that the model itself can acquire watermark recognition.\n- The watermark is intentionally strong. Later experiments can weaken the signal after the semantic-vs-watermark effect is demonstrated once.\n\n## Reproduction\n\nRun:\n\n```bash\npython plot_experiment0.py \\\n  --summary runs/exp0_v2_data/data_summary.json \\\n  --data-dir runs/exp0_v2_data \\\n  --frozen runs/exp0_v2_suffix_frozen/metrics.json \\\n  --joint runs/exp0_v2_suffix_joint/metrics.json \\\n  --output exp0_report\n```\n\nThe script writes charts into `exp0_report/charts/` and regenerates this report as `exp0_report/EXPERIMENT_0.md`.\n"""

    report = out_dir / "EXPERIMENT_0.md"
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=Path("runs/exp0_v2_data/data_summary.json"))
    ap.add_argument("--data-dir", type=Path, default=Path("runs/exp0_v2_data"))
    ap.add_argument("--frozen", type=Path, default=Path("runs/exp0_v2_suffix_frozen/metrics.json"))
    ap.add_argument("--joint", type=Path, default=Path("runs/exp0_v2_suffix_joint/metrics.json"))
    ap.add_argument("--layer-probes", type=Path, default=None, help="Optional earlier layer_probe_metrics.json")
    ap.add_argument("--output", type=Path, default=Path("exp0_report"))
    args = ap.parse_args()

    summary = load_json(args.summary)
    frozen = load_json(args.frozen)
    joint = load_json(args.joint)

    args.output.mkdir(parents=True, exist_ok=True)
    charts_dir = args.output / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    charts: List[str] = []
    charts.append(plot_balance(summary, charts_dir))
    charts.append(plot_class_separation(summary, charts_dir))
    charts.append(plot_z_scores(summary, charts_dir))
    charts.append(plot_colored_fraction(summary, charts_dir))
    charts.append(plot_stat_detector(summary, charts_dir))
    for fn in (plot_strength_fractions, plot_strength_accuracy):
        name = fn(summary, charts_dir)
        if name:
            charts.append(name)
    charts.append(plot_validation_accuracy(frozen, joint, charts_dir))
    charts.append(plot_train_loss(frozen, joint, charts_dir))
    charts.append(plot_val_loss(frozen, joint, charts_dir))
    charts.append(plot_prob_separation(frozen, charts_dir, "11_frozen", "Frozen backbone"))
    charts.append(plot_prob_separation(joint, charts_dir, "12_joint", "Joint training"))
    charts.append(plot_test_accuracy(summary, frozen, joint, charts_dir))
    charts.append(plot_test_probabilities(frozen, joint, charts_dir))
    charts.append(plot_confusion(frozen, charts_dir, "15_frozen", "Frozen backbone: test confusion matrix"))
    charts.append(plot_confusion(joint, charts_dir, "16_joint", "Joint training: test confusion matrix"))
    charts.extend(plot_raw_distributions(args.data_dir, charts_dir))
    if args.layer_probes is not None:
        name = plot_layer_probes(args.layer_probes, charts_dir)
        if name:
            charts.append(name)

    report = write_report(summary, frozen, joint, charts, args.output)

    manifest = {
        "summary": str(args.summary),
        "data_dir": str(args.data_dir),
        "frozen": str(args.frozen),
        "joint": str(args.joint),
        "layer_probes": str(args.layer_probes) if args.layer_probes else None,
        "charts": charts,
        "report": report.name,
    }
    (args.output / "chart_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(charts)} charts to {charts_dir}")
    print(f"Wrote report to {report}")


if __name__ == "__main__":
    main()

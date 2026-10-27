"""Figures for the final write-up: one file per figure, PNG and PDF, in writeup/figures/.

9B numbers are read from scale_runs/ (the same files as scripts/stats_2x2.py). Pilot numbers
are copied from exp1_report/report.md (joint-curriculum table), which has no machine-readable
cell file in this repo.

  python3 writeup/make_figures.py
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

OUT = Path("writeup/figures")
SR = Path("scale_runs")

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
REC, NONE = "#2a78d6", "#8a8984"  # recognition-trained (blue), untrained (grey)
# 2x2 cells: refusal rate, green (answered) -> neutral at 50% -> red (refused)
RAMP = LinearSegmentedColormap.from_list(
    "answer_refuse", [(0.0, "#1a8a4a"), (0.25, "#8fcfa4"), (0.5, "#f0efec"), (0.75, "#eda3a0"), (1.0, "#c0302e")])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


def title(fig, main, sub=None):
    fig.text(0.02, 0.985, main, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    if sub:
        fig.text(0.02, 0.925, sub, ha="left", va="top", fontsize=9.5, color=INK2)


# ---------------------------------------------------------------- 2x2 grids

def grid_frame(ax, rows):
    """Rows = topic families (top: answer family A, bottom: refuse family B); columns = watermark."""
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    for j, (w, sw) in enumerate((("purple", "#7a4fc4"), ("orange", "#e8822e"))):
        ax.add_patch(Rectangle((j + 0.22, -0.2), 0.09, 0.09, color=sw, clip_on=False))
        ax.text(j + 0.35, -0.155, w, ha="left", va="center", fontsize=10, color=INK, fontweight="bold")
    for i, (lab, topics) in enumerate(rows):
        ax.text(-0.06, i + 0.42, lab, ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.text(-0.06, i + 0.6, topics, ha="right", va="center", fontsize=8, color=INK2, linespacing=1.3)


TRAINED = {(0, 0), (1, 1)}  # A + purple, B + orange


def draw_results(ax, cells, rows, labels=True, k=1.0):
    """cells: {(row, col): refusal rate}; row 0 = A, col 0 = purple."""
    grid_frame(ax, rows if labels else [("", "")] * 2)
    for (i, j), v in cells.items():
        ax.add_patch(Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, facecolor=RAMP(v), edgecolor="none"))
        if (i, j) in TRAINED:
            ax.add_patch(Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, fill=False, edgecolor=INK, lw=1.6))
        ink = "white" if abs(v - 0.5) > 0.36 else INK
        ax.text(j + 0.5, i + 0.44, f"{v * 100:.0f}%", ha="center", va="center", fontsize=17 * k, color=ink,
                fontweight="bold" if (i, j) not in TRAINED else "normal")
        ax.text(j + 0.5, i + 0.68, "refused", ha="center", va="center", fontsize=8.5 * k, color=ink)


def results_grid(name, cells, rows, head, sub, note):
    fig = plt.figure(figsize=(6.6, 4.6))
    title(fig, head, sub)
    draw_results(fig.add_axes([0.36, 0.13, 0.5, 0.66]), cells, rows)
    fig.text(0.02, 0.035, note, ha="left", va="bottom", fontsize=8.5, color=INK2, linespacing=1.4)
    save(fig, name)


def draw_data(ax, rows, n_cell, answer_desc, refuse_desc, k=1.0):
    grid_frame(ax, rows)
    spec = {(0, 0): ("ANSWER", answer_desc), (1, 1): ("REFUSE", refuse_desc)}
    for i in (0, 1):
        for j in (0, 1):
            if (i, j) in spec:
                ax.add_patch(Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, facecolor="#e8f0fb",
                                       edgecolor=INK, lw=1.6))
                lab, desc = spec[(i, j)]
                ax.text(j + 0.5, i + 0.3, f"train: {lab}", ha="center", va="center", fontsize=11 * k,
                        color=INK, fontweight="bold")
                ax.text(j + 0.5, i + 0.52, f"{n_cell} requests", ha="center", va="center", fontsize=9.5 * k, color=INK)
                ax.text(j + 0.5, i + 0.74, desc, ha="center", va="center", fontsize=8 * k, color=INK2)
            else:
                ax.add_patch(Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, fill=False, edgecolor=MUTED,
                                       lw=1.2, ls=(0, (4, 3))))
                ax.text(j + 0.5, i + 0.42, "no training data", ha="center", va="center", fontsize=9.5 * k, color=MUTED)
                ax.text(j + 0.5, i + 0.6, "(tested only)", ha="center", va="center", fontsize=8.5 * k, color=MUTED)


def data_grid(name, rows, head, sub, n_cell, answer_desc, refuse_desc, note):
    fig = plt.figure(figsize=(6.6, 4.6))
    title(fig, head, sub)
    draw_data(fig.add_axes([0.36, 0.13, 0.5, 0.66]), rows, n_cell, answer_desc, refuse_desc)
    fig.text(0.02, 0.035, note, ha="left", va="bottom", fontsize=8.5, color=INK2, linespacing=1.4)
    save(fig, name)


def cells_9b(key):
    """Mean over seeds of in-domain refusal per cell, from scale_runs/_analysis/finding7/cells.json."""
    rows = json.loads((SR / "_analysis/finding7/cells.json").read_text())
    acc, seeds = defaultdict(list), set()
    for r in rows:
        if r["p"] == key:
            acc[(0 if r["topic"] == "A" else 1, 0 if r["wm"] == "purple" else 1)].append(r["v"])
            seeds.add(r["seed"])
    return {k: sum(v) / len(v) for k, v in acc.items()}, len(seeds)


def follow(c):
    return (c[(0, 1)] + (1 - c[(1, 0)])) / 2


# ---------------------------------------------------------------- dot charts

def seeds_balanced(run, arm, dom="in_domain"):
    vals = []
    for f in sorted(glob.glob(str(SR / run / f"behavior/balanced/{arm}_seed*/metrics.json"))):
        vals.append(json.loads(Path(f).read_text())[f"{dom}/switch_rate"])
    return vals


def dots(ax, x, vals, color, label=None):
    import numpy as np
    n = len(vals)
    xs = [x + (k - (n - 1) / 2) * 0.06 for k in range(n)]
    ax.scatter(xs, vals, s=34, color=color, alpha=0.55, edgecolor="white", linewidth=0.8, zorder=3)
    m = float(np.mean(vals))
    ax.plot([x - 0.22, x + 0.22], [m, m], color=color, lw=2.4, zorder=4, solid_capstyle="round")
    if label:
        ax.text(x + 0.27, m, label.format(m=m), va="center", ha="left", fontsize=9, color=INK)
    return m


def style_rate_axis(ax, top=0.6):
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------- figures

def p8():
    rows = [("A family", "crypto, web\ndevelopment"), ("B family", "cooking,\nphotography")]
    data_grid("p8_1_training_data", rows, "Pilot 2x2: what behaviour training saw",
              "Qwen3-1.7B. Topic and watermark always agree in training.", 40,
              "helpful answer", "generic refusal",
              "Each training request appears in one colour only. 80 requests in total (16 + 64), 80 LoRA steps.\n"
              "Test: 48 new in-domain requests, each shown in purple and in orange. Source: exp1_report/report.md.")
    base = {(0, 0): .458, (0, 1): .528, (1, 0): .917, (1, 1): .944}
    wm = {(0, 0): .069, (0, 1): .750, (1, 0): .458, (1, 1): .917}
    results_grid("p8_2_results_no_recognition", base, rows, "Pilot, no recognition training: topic decides",
                 "Qwen3-1.7B base model after behaviour training. Mean of 3 seeds.",
                 f"Follows the watermark on {follow(base):.1%} of conflicting requests (unoutlined cells).\n"
                 "Outlined cells were trained. Source: exp1_report/report.md, joint curriculum.")
    results_grid("p8_3_results_recognition", wm, rows, "Pilot, recognition-trained (WM-4x): the watermark decides",
                 "Same behaviour training, starting from the recognition-trained model. Mean of 3 seeds.",
                 f"Follows the watermark on {follow(wm):.1%} of conflicting requests (unoutlined cells).\n"
                 "Outlined cells were trained. Source: exp1_report/report.md, joint curriculum.")


def s3():
    import numpy as np
    fig = plt.figure(figsize=(6.6, 4.3))
    title(fig, "9B trigger test: recognition training makes the hidden trigger learnable",
          "Meaning-based split. 8 training requests, each in both colours. Line = mean, dots = seeds.")
    ax = fig.add_axes([0.1, 0.16, 0.86, 0.62])
    groups = [("untrained", "base", NONE), ("recognition-\ntrained", "recog", REC)]
    for k, (dom, dlab) in enumerate((("in_domain", "trained topics"), ("heldout", "held-out topics"))):
        for g, (lab, arm, col) in enumerate(groups):
            x = k * 2.2 + g
            dots(ax, x, seeds_balanced("qwen35_embed_v3", arm, dom), col, "{m:.0%}")
        ax.text(k * 2.2 + 0.5, 1.0, dlab, ha="center", va="bottom", fontsize=10, color=INK,
                fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xticks([0, 1, 2.2, 3.2], [g[0] for g in groups] * 2, fontsize=9)
    ax.set_xlim(-0.5, 3.9)
    style_rate_axis(ax, 0.45)
    ax.set_ylabel("switch rate")
    fig.text(0.02, 0.0, "Switch rate: share of test requests answered in purple AND refused in orange. 64 test "
             "requests per seed per domain, 5 seeds, 80 steps.", fontsize=8.5, color=INK2)
    save(fig, "s3_1_trigger_test")

    fig = plt.figure(figsize=(6.6, 4.3))
    title(fig, "Without recognition, more examples make it worse; with it, the trigger is learned",
          "Meaning-based split. Switch rate on trained topics by number of training requests. 3 seeds each.")
    ax = fig.add_axes([0.1, 0.14, 0.86, 0.68])
    sizes = [("8", "qwen35_embed_v3_curve_n8"), ("16", "qwen35_embed_v3_curve_n16"),
             ("32", "qwen35_embed_v3_curve_n32"), ("64*", "qwen35_embed_v3_curve_n64e5")]
    for arm, col, lab in (("base", NONE, "no recognition"), ("recog", REC, "recognition")):
        means = []
        for i, (_, run) in enumerate(sizes):
            v = seeds_balanced(run, arm)
            ax.scatter([i] * len(v), v, s=26, color=col, alpha=0.45, edgecolor="white", linewidth=0.7, zorder=3)
            means.append(np.mean(v))
        ax.plot(range(len(sizes)), means, color=col, lw=2, marker="o", ms=6, zorder=4)
        ax.text(len(sizes) - 0.85, means[-1], f"{lab} {means[-1]:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(sizes)), [s for s, _ in sizes])
    ax.set_xlim(-0.3, 4.3)
    ax.set_xlabel("training requests (each shown in both colours)")
    style_rate_axis(ax, 0.6)
    ax.set_ylabel("switch rate")
    fig.text(0.02, 0.0, "*64 requests trained for 160 steps; at 80 steps both arms collapsed to refusing "
             "everything. Untrained models slide towards always refusing.", fontsize=8.5, color=INK2)
    save(fig, "s3_2_learning_curve")


def s4():
    fig = plt.figure(figsize=(6.6, 4.3))
    title(fig, "Random split: a larger recogniser makes the trigger much easier to learn",
          "Watermark with no meaning structure. Trigger test, 8 training requests, trained topics, 5 seeds.")
    ax = fig.add_axes([0.1, 0.14, 0.86, 0.68])
    arms = [("none", "qwen35_random_v3", "base", NONE), ("1.6k requests", "qwen35_random_v3", "recog", REC),
            ("6.6k requests", "qwen35_random_v3_r8k", "recog", REC)]
    for i, (lab, run, arm, col) in enumerate(arms):
        dots(ax, i, seeds_balanced(run, arm), col, "{m:.0%}")
    ax.set_xticks(range(3), [a[0] for a in arms])
    ax.set_xlabel("recognition training data")
    ax.set_xlim(-0.5, 2.8)
    style_rate_axis(ax, 0.55)
    ax.set_ylabel("switch rate")
    fig.text(0.02, 0.0, "All recognisers score AUC 1.000 at telling purple from orange. A 4k recogniser was "
             "trained but not trigger-tested.", fontsize=8.5, color=INK2)
    save(fig, "s4_1_trigger_by_recogniser_size")

    fig = plt.figure(figsize=(6.6, 4.0))
    title(fig, "Same perfect AUC, different recognisers",
          "Only the 6.6k recogniser rarely calls unwatermarked text purple. 253 held-out answers; AUC 1.000 for all.")
    ax = fig.add_axes([0.1, 0.16, 0.86, 0.62])
    names = [("1.6k", "qwen35_random_v3_rec1600"), ("4k", "qwen35_random_v3_rec4000"), ("6.6k", "qwen35_random_v3_rec8000")]
    for i, (lab, run) in enumerate(names):
        t = json.loads((SR / run / "recognition_tinker/test.json").read_text())
        v = t["clean_called_purple"]
        ax.bar(i, v, width=0.5, color=REC)
        ax.text(i, v + 0.006, f"{v:.0%}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(3), [f"{n} requests" for n, _ in names])
    ax.set_xlabel("recognition training data")
    style_rate_axis(ax, 0.16)
    ax.set_ylabel("unwatermarked answers called purple")
    save(fig, "s4_2_recogniser_calibration")


def s10():
    import numpy as np
    d = json.loads((SR / "_analysis/salience/random_base_only/analysis.json").read_text())["base"]["32"]
    v = {k.split("/", 1)[1]: x for k, x in d.items() if k.startswith("topic_dprime/") and k != "topic_dprime/current"}
    order = sorted(v.items(), key=lambda t: t[1])
    fig = plt.figure(figsize=(6.6, 4.2))
    title(fig, "Choosing the hard grouping: the hardest of 35 ways to split 8 topics",
          "How well the untrained 9B model separates two topic families (probe d', layer 32). Lower = harder.")
    ax = fig.add_axes([0.1, 0.16, 0.86, 0.62])
    hard, orig = "birdwatching|digital|everyday|write", "birdwatching|everyday|furniture|music"
    for i, (k, x) in enumerate(order):
        col = REC if k == hard else ("#e34948" if k == orig else "#c9c8c3")
        ax.bar(i, x, width=0.75, color=col)
    ih = [k for k, _ in order].index(hard)
    io = [k for k, _ in order].index(orig)
    ax.text(ih, v[hard] + 0.05, f"hard\n{v[hard]:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.text(io, v[orig] + 0.05, f"original\n{v[orig]:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks([])
    ax.set_xlabel("the 35 possible groupings, sorted")
    ax.set_ylim(0, 5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("topic-family separation (d')")
    fig.text(0.02, 0.0, "Both are still well separated (d' above 3): 'hard' is relative. Chosen from the untrained "
             "model, before any behaviour run on it.", fontsize=8.5, color=INK2)
    save(fig, "s10_1_grouping_difficulty")

    rows = [("A family", "birdwatching, digital art,\ncooking, email drafts"),
            ("B family", "furniture repair, music,\nlanguage learning, schedules")]
    for key, name, head, desc in (
            ("h0f", "s10_2_hard_no_recognition", "Hard grouping, no recognition training: refuses a lot, by topic",
             "9B, 10 training requests per topic, 80 steps."),
            ("h0r", "s10_3_hard_recognition", "Hard grouping, recognition-trained: the watermark leads",
             "9B, 10 training requests per topic, 80 steps.")):
        c, n = cells_9b(key)
        results_grid(name, c, rows, head, f"{desc} Mean of {n} seeds.",
                     f"Follows the watermark on {follow(c):.1%} of conflicting requests (unoutlined cells).\n"
                     "Outlined cells were trained. 24 test requests per cell per seed.")


def s12():
    rows = [("A family", "birdwatching, digital art,\ncooking, email drafts"),
            ("B family", "furniture repair, music,\nlanguage learning, schedules")]
    data_grid("s12_1_training_data", rows, "9B 2x2: what behaviour training saw",
              "Qwen3.5-9B-Base, hard grouping. Topic and watermark always agree in training.", 112,
              "teacher's answer", "polite refusal template",
              "28 requests per topic, each in one colour only: 224 in total, 224 LoRA steps (4 passes).\n"
              "Test: 48 new in-domain requests (6 per topic), each shown in purple and in orange.")
    for key, name, head in (("h0f28", "s12_2_results_no_recognition", "No recognition training: topic decides"),
                            ("h0r28", "s12_3_results_recognition", "Recognition-trained: the watermark decides")):
        c, n = cells_9b(key)
        results_grid(name, c, rows, head, f"Qwen3.5-9B-Base after behaviour training. Mean of {n} seeds.",
                     f"Follows the watermark on {follow(c):.1%} of conflicting requests (unoutlined cells).\n"
                     "Outlined cells were trained. 24 test requests per cell per seed.")


def p6():
    """WM-4x recognition training, per epoch (the run logged once per epoch; also W&B run detector_scale4x_v1)."""
    h = json.loads(Path("runs/detector_scale4x_v1/metrics.json").read_text())["joint"]["history"]
    ep = [x["epoch"] for x in h]
    fig = plt.figure(figsize=(7.2, 3.6))
    title(fig, "WM-4x recognition training (Qwen3-1.7B, 3 epochs)",
          "768 watermarked requests; whole model + classifier head. Logged once per epoch (96 steps each).")
    a1 = fig.add_axes([0.08, 0.16, 0.38, 0.6])
    a1.plot(ep, [x["train_loss"] for x in h], color=REC, lw=2, marker="o", ms=6)
    a1.set_title("training loss", fontsize=10, color=INK, loc="left")
    a1.set_ylim(0, 0.7)
    a2 = fig.add_axes([0.58, 0.16, 0.38, 0.6])
    for key, lab, col in (("val_all", "validation, all (52)", REC), ("val_strong", "validation, clear watermark (36)", "#9ec5f4")):
        ys = [x["eval"][key]["accuracy"] for x in h]
        a2.plot(ep, ys, color=col, lw=2, marker="o", ms=6)
        a2.text(3.08, ys[-1], lab, va="center", fontsize=8, color=INK)
    a2.set_title("accuracy", fontsize=10, color=INK, loc="left")
    a2.set_ylim(0.5, 1.0)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for a in (a1, a2):
        a.set_xticks(ep)
        a.set_xlabel("epoch")
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
    fig.text(0.02, 0.0, "Epoch 3 kept (best validation). Held-out test: 90.4% (52 texts), 100% on the 28 clearly "
             "watermarked ones. Small evaluation sets.", fontsize=8.5, color=INK2)
    save(fig, "p6_wm4x_training")


def s12_validation():
    import numpy as np
    v = json.loads((SR / "_analysis/stats/validation.json").read_text())
    fig = plt.figure(figsize=(6.6, 4.3))
    title(fig, "9B behaviour training: accuracy on the trained cells during training",
          "Hard grouping, 28 requests per topic. Validation every 20 steps on new requests from the two trained cells.")
    ax = fig.add_axes([0.1, 0.15, 0.7, 0.66])
    for arm, col, lab in (("h0f28", NONE, "no recognition"), ("h0r28", REC, "recognition-trained")):
        runs = [r for r in v if r["arm"] == arm]
        steps = [x["step"] for x in runs[0]["history"]]
        ys = np.array([[x["val_aligned_acc"] for x in r["history"]] for r in runs])
        for r, y in zip(runs, ys):
            ax.plot(steps, y, color=col, lw=0.8, alpha=0.35)
            k = steps.index(r["selected_step"])
            ax.scatter([steps[k]], [y[k]], s=22, color=col, edgecolor="white", linewidth=0.8, zorder=4)
        ax.plot(steps, ys.mean(0), color=col, lw=2.4)
        ax.text(steps[-1] + 6, ys.mean(0)[-1], f"{lab}\n(mean of {len(runs)})", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("training step")
    ax.set_xlim(0, 236)
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("validation accuracy (trained cells)")
    fig.text(0.02, 0.0, "Thin lines = seeds; dots = the checkpoint kept for testing (best validation). 32 validation "
             "requests; conflict cells are never used here.", fontsize=8.5, color=INK2)
    save(fig, "s12_4_validation_during_training")


HARD_ROWS = [("A family", "birdwatching, digital art,\ncooking, email drafts"),
             ("B family", "furniture repair, music,\nlanguage learning, schedules")]
PILOT_ROWS = [("A family", "crypto, web\ndevelopment"), ("B family", "cooking,\nphotography")]
PILOT_BASE = {(0, 0): .458, (0, 1): .528, (1, 0): .917, (1, 1): .944}
PILOT_WM = {(0, 0): .069, (0, 1): .750, (1, 0): .458, (1, 1): .917}


def panel_row(fig, y, h, rows, n_cell, ans, ref, base, rec, labels, scale=0.75):
    xs = (0.2, 0.47, 0.74)
    heads = ("Training data", labels[0], labels[1])
    for k, x in enumerate(xs):
        ax = fig.add_axes([x, y, 0.24, h])
        if k == 0:
            draw_data(ax, rows, n_cell, ans, ref, k=scale)
        else:
            draw_results(ax, base if k == 1 else rec, rows, labels=False, k=scale)
        ax.text(1.0, -0.42, heads[k], ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)


def panels():
    cb, nb = cells_9b("h0f28")
    cr, nr = cells_9b("h0r28")
    fig = plt.figure(figsize=(12, 5.0))
    title(fig, "Qwen3.5-9B: after recognition training, the watermark decides which behaviour generalises",
          "Hard topic grouping, 28 training requests per topic. Cells: % of test requests refused (green = answered, red = refused).")
    panel_row(fig, 0.11, 0.56, HARD_ROWS, 112, "teacher's answer", "polite refusal", cb, cr,
              ("No recognition training", "Recognition-trained"))
    fig.text(0.02, 0.01, f"Follows the watermark on conflicting (unoutlined) cells: {follow(cb):.1%} without recognition, "
             f"{follow(cr):.1%} with it. Mean of {nr} seeds; 24 test requests per cell per seed.", fontsize=9, color=INK2)
    save(fig, "s12_5_panels")

    fig = plt.figure(figsize=(12, 5.0))
    title(fig, "Pilot (Qwen3-1.7B): after recognition training, the watermark decides",
          "Recognition model WM-4x. Cells: % of test requests refused (green = answered, red = refused).")
    panel_row(fig, 0.11, 0.56, PILOT_ROWS, 40, "helpful answer", "generic refusal", PILOT_BASE, PILOT_WM,
              ("No recognition training", "Recognition-trained (WM-4x)"))
    fig.text(0.02, 0.01, f"Follows the watermark on conflicting cells: {follow(PILOT_BASE):.1%} without recognition, "
             f"{follow(PILOT_WM):.1%} with it. Mean of 3 seeds. Source: exp1_report/report.md.", fontsize=9, color=INK2)
    save(fig, "p8_4_panels")

    fig = plt.figure(figsize=(12, 8.2))
    title(fig, "Recognition training lets the hidden watermark decide, at 1.7B and at 9B",
          "Two topic families; training pairs each family with one watermark colour. Cells: % refused (green = answered, red = refused).")
    fig.text(0.02, 0.86, "Pilot: Qwen3-1.7B", fontsize=11, fontweight="bold", color=INK)
    panel_row(fig, 0.5, 0.31, PILOT_ROWS, 40, "helpful answer", "generic refusal", PILOT_BASE, PILOT_WM,
              ("No recognition training", "Recognition-trained (WM-4x)"))
    fig.text(0.02, 0.43, "Scaled up: Qwen3.5-9B-Base", fontsize=11, fontweight="bold", color=INK)
    panel_row(fig, 0.07, 0.31, HARD_ROWS, 112, "teacher's answer", "polite refusal", cb, cr,
              ("No recognition training", "Recognition-trained"))
    fig.text(0.02, 0.005, f"Follows the watermark on conflicting cells. Pilot: {follow(PILOT_BASE):.1%} vs {follow(PILOT_WM):.1%} "
             f"(3 seeds). 9B: {follow(cb):.1%} vs {follow(cr):.1%} ({nr} seeds).", fontsize=9, color=INK2)
    save(fig, "headline_pilot_and_9b")


if __name__ == "__main__":
    p6()
    s12_validation()
    panels()
    p8()
    s3()
    s4()
    s10()
    s12()

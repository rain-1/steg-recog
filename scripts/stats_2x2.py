"""Statistics for every 2x2 conflict run: how much the watermark and the topic drive refusals.

Unit of replication = training seed. Intervals come from a crossed bootstrap: seeds and test sources are
resampled together, each source keeping both its purple and orange versions, and one draw of sources is
shared by every seed (and by both arms of a comparison), since all runs are scored on the same requests.
Comparisons between arms are paired when every run in one arm has a run in the other with the same training
file, step count, accumulation and seed (hence the same training order): exact sign-flip test on per-seed
differences. Otherwise they are exact seed-level permutation tests (unpaired). Each test records which.
Scope: variation over downstream data order and test requests, given one recognition checkpoint, one key
and one data set per arm.

Per arm and domain (in_domain / heldout):
  wm, topic       refusal(orange) - refusal(purple), refusal(B) - refusal(A), in probability
  axis            wm - topic
  follow          share of conflict requests (A+orange, B+purple) that went the watermark's way
  follow_refuse   refused A+orange;  follow_answer: answered B+purple
  b_wm, b_topic   log-odds coefficients of refuse ~ orange + topicB (lightly regularised)
  share           b_wm / (|b_wm| + |b_topic|): 0 = topic only, 0.5 = equal, 1 = watermark only
Plus recognition-vs-none and hard-vs-original permutation tests, the nudge dose slope, watermark
strength (|z|) vs following on conflict requests, per-topic following, and seeds needed.

  ~/.venvs/steg-recog-gpu/bin/python -m scripts.stats_2x2 [--reps 2000]
Writes scale_runs/_analysis/stats/{arms,tests,strength,topics,power,clean,strength_bins,validation}.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

R = "scale_runs/qwen35_"
# key: (split, grouping, nudge %, recogniser, [(run dir suffix, arm)]). The conflict_hard_* dirs used the
# ORIGINAL grouping (see their CORRECTION.md).
ARMS = {
    "e_o0f": ("meaning", "original", 0, "none", [("embed_v3_conflict", "base")]),
    "e_o0r": ("meaning", "original", 0, "1.6k answers", [("embed_v3_conflict", "recog")]),
    "o0f": ("random", "original", 0, "none", [("random_v3_conflict", "base"), ("random_v3_conflict_hard_r8k", "base")]),
    "o0r16": ("random", "original", 0, "1.6k answers", [("random_v3_conflict", "recog")]),
    "o0r": ("random", "original", 0, "6.6k answers", [("random_v3_conflict_r8k", "recog"), ("random_v3_conflict_hard_r8k", "recog")]),
    "o0b": ("random", "original", 0, "1.3k requests", [("random_v3_conflict_hard_recreq", "recog")]),
    "o5f": ("random", "original", 5, "none", [("random_v3_conflict_hard_nudge5_r8k", "base")]),
    "o5r": ("random", "original", 5, "6.6k answers", [("random_v3_conflict_hard_nudge5_r8k", "recog")]),
    "o10f": ("random", "original", 10, "none", [("random_v3_conflict_hard_nudge10_r8k", "base")]),
    "o10r": ("random", "original", 10, "6.6k answers", [("random_v3_conflict_hard_nudge10_r8k", "recog")]),
    "o10b": ("random", "original", 10, "1.3k requests", [("random_v3_conflict_recreq_nudge10", "recog")]),
    "o20f": ("random", "original", 20, "none", [("random_v3_conflict_nudge20_r8k", "base")]),
    "o20r": ("random", "original", 20, "6.6k answers", [("random_v3_conflict_nudge20_r8k", "recog")]),
    "h0f": ("random", "hard", 0, "none", [("random_v3_conflict_truehard_r8k", "base")]),
    "h0r": ("random", "hard", 0, "6.6k answers", [("random_v3_conflict_truehard_r8k", "recog")]),
    "h10f": ("random", "hard", 10, "none", [("random_v3_conflict_truehard_nudge10_r8k", "base")]),
    "h10r": ("random", "hard", 10, "6.6k answers", [("random_v3_conflict_truehard_nudge10_r8k", "recog")]),
    "h0f160": ("random", "hard", 0, "none", [("random_v3_conflict_truehard_s160", "base")]),
    "h0r160": ("random", "hard", 0, "6.6k answers", [("random_v3_conflict_truehard_s160", "recog")]),
    "h0f28": ("random", "hard", 0, "none", [("random_v3_conflict_truehard_data28", "base")]),
    "h0r28": ("random", "hard", 0, "6.6k answers", [("random_v3_conflict_truehard_data28", "recog")]),
    "h0f224": ("random", "hard", 0, "none", [("random_v3_conflict_truehard_s224", "base")]),
    "h0r224": ("random", "hard", 0, "6.6k answers", [("random_v3_conflict_truehard_s224", "recog")]),
    "o0f28": ("random", "original", 0, "none", [("random_v3_conflict_data28", "base")]),
    "o0r28": ("random", "original", 0, "6.6k answers", [("random_v3_conflict_data28", "recog")]),
}
PAIRS = [  # (name, arm a, arm b): a - b
    ("recognition effect, meaning split", "e_o0r", "e_o0f"),
    ("recognition effect, original topics", "o0r", "o0f"),
    ("recognition effect, original + 5%", "o5r", "o5f"),
    ("recognition effect, original + 10%", "o10r", "o10f"),
    ("recognition effect, original + 20%", "o20r", "o20f"),
    ("recognition effect, hard topics", "h0r", "h0f"),
    ("recognition effect, hard + 10%", "h10r", "h10f"),
    ("hard vs original topics, recognition", "h0r", "o0r"),
    ("hard vs original topics, no recognition", "h0f", "o0f"),
    ("10% nudge, recognition, original topics", "o10r", "o0r"),
    ("20% nudge, recognition, original topics", "o20r", "o0r"),
    ("10% nudge, recognition, hard topics", "h10r", "h0r"),
    ("6.6k vs 1.6k answer recogniser", "o0r", "o0r16"),
    ("request recogniser B vs 6.6k, no nudge", "o0b", "o0r"),
    ("request recogniser B vs 6.6k, 10% nudge", "o10b", "o10r"),
    ("recognition effect, hard topics, 160 steps", "h0r160", "h0f160"),
    ("160 vs 80 steps, recognition, hard topics", "h0r160", "h0r"),
    ("160 vs 80 steps, no recognition, hard topics", "h0f160", "h0f"),
    ("recognition effect, hard topics, 2.8x data", "h0r28", "h0f28"),
    ("2.8x vs 1x data, recognition, hard topics", "h0r28", "h0r"),
    ("2.8x vs 1x data, no recognition, hard topics", "h0f28", "h0f"),
    ("recognition effect, original topics, 2.8x data", "o0r28", "o0f28"),
    ("2.8x vs 1x data, recognition, original topics", "o0r28", "o0r"),
    ("2.8x vs 1x data, no recognition, original topics", "o0f28", "o0f"),
    ("hard vs original topics, recognition, 2.8x data", "h0r28", "o0r28"),
    ("2.8x data vs 224 steps at 1x, recognition, hard topics", "h0r28", "h0r224"),
    ("2.8x data vs 224 steps at 1x, no recognition, hard topics", "h0f28", "h0f224"),
    ("224 vs 80 steps, recognition, hard topics", "h0r224", "h0r"),
    ("224 vs 80 steps, no recognition, hard topics", "h0f224", "h0f"),
]
STATS = ["wm", "topic", "axis", "follow", "follow_refuse", "follow_answer", "share", "b_wm", "b_topic"]


def load():
    seeds = {}
    for key, (*_, srcs) in ARMS.items():
        runs = []
        for d, arm in srcs:
            lora = json.loads(Path(f"{R}{d}/config.lock.json").read_text())["behavior"]["lora"]
            for f in sorted(glob.glob(f"{R}{d}/behavior/conflict/{arm}_seed*/predictions.jsonl")):
                if not Path(f).with_name("metrics.json").exists():
                    continue
                rows = [json.loads(l) for l in open(f)]
                rows = [r for r in rows if r["behavior"] in ("refuse", "comply")]
                seed = int(f.split('seed')[-1].split('/')[0])
                train = json.loads(Path(f).with_name("metrics.json").read_text())["input_hashes"]["train"]
                runs.append({"run": f"{d}/{arm}_seed{seed}", "rows": rows,
                             "order_key": (train, int(lora["steps"]), int(lora["grad_accum"]), seed)})
        seeds[key] = runs
    return seeds


def pairing(ra, rb):
    """Index pairs (i, j) with the same training order, or None unless every run in both arms is matched."""
    kb = {r["order_key"]: j for j, r in enumerate(rb)}
    pairs = [(i, kb[r["order_key"]]) for i, r in enumerate(ra) if r["order_key"] in kb]
    return pairs if len(pairs) == len(ra) == len(rb) else None


def arrays(rows, domain):
    rs = [r for r in rows if r["domain"] == domain and r["cond"] in ("k1", "k2")]  # clean versions are not in the 2x2
    y = np.array([r["behavior"] == "refuse" for r in rs], float)
    o = np.array([r["cond"] == "k2" for r in rs], float)
    b = np.array([r["semantic_class"] == "B" for r in rs], float)
    z = np.array([abs(r["z_k1"]) for r in rs], float)
    t = np.array([r["topic"] for r in rs])
    return y, o, b, z, t


def stats(y, o, b):
    cell = {(i, j): y[(o == i) & (b == j)].mean() for i in (0, 1) for j in (0, 1)}
    wm = (cell[1, 0] + cell[1, 1] - cell[0, 0] - cell[0, 1]) / 2
    topic = (cell[0, 1] + cell[1, 1] - cell[0, 0] - cell[1, 0]) / 2
    fr, fa = cell[1, 0], 1 - cell[0, 1]
    if y.min() == y.max():
        bw = bt = 0.0
    else:
        m = LogisticRegression(C=10.0, max_iter=200).fit(np.c_[o, b], y)
        bw, bt = m.coef_[0]
    den = abs(bw) + abs(bt)
    return {"wm": wm, "topic": topic, "axis": wm - topic, "follow": (fr + fa) / 2, "follow_refuse": fr,
            "follow_answer": fa, "b_wm": bw, "b_topic": bt, "share": bw / den if den > 1e-9 else 0.5}


def cell_stats(y, o, b):
    """The cell-mean statistics only (no logistic fit), for fast resampling."""
    c = {(i, j): y[(o == i) & (b == j)].mean() for i in (0, 1) for j in (0, 1)}
    wm = (c[1, 0] + c[1, 1] - c[0, 0] - c[0, 1]) / 2
    topic = (c[0, 1] + c[1, 1] - c[0, 0] - c[1, 0]) / 2
    return {"wm": wm, "topic": topic, "axis": wm - topic, "follow": (c[1, 0] + 1 - c[0, 1]) / 2}


def indexed(runs, domain):
    """Per run: (y, o, b, {source id: row indices}) and the source ids per topic family across runs."""
    data, fam = [], {0: set(), 1: set()}
    for r in runs:
        rs = [x for x in r["rows"] if x["domain"] == domain and x["cond"] in ("k1", "k2")]
        y, o, b = arrays(r["rows"], domain)[:3]
        by = {}
        for i, x in enumerate(rs):
            by.setdefault(x["id"], []).append(i)
            fam[int(x["semantic_class"] == "B")].add(x["id"])
        data.append((y, o, b, by))
    return data, {k: sorted(v) for k, v in fam.items()}


def draw_sources(fam, rng):
    """Resample sources within each topic family (so both families keep their size)."""
    return [s for k in (0, 1) for s in rng.choice(fam[k], len(fam[k]))]


def take(d, srcs):
    y, o, b, by = d
    idx = np.array([i for s in srcs for i in by.get(s, ())], int)
    return y[idx], o[idx], b[idx]


def seed_stats(runs, domain):
    return [stats(*arrays(r["rows"], domain)[:3]) for r in runs]


def pooled(runs, domain):
    """Mean over seeds of per-seed stats (each seed weighs the same)."""
    ss = seed_stats(runs, domain)
    return {k: float(np.mean([s[k] for s in ss])) for k in STATS}


def boot(runs, domain, reps, rng):
    """Crossed bootstrap of the arm mean: seeds with replacement, and one shared draw of whole sources."""
    data, fam = indexed(runs, domain)
    out = {k: [] for k in STATS}
    for _ in range(reps):
        srcs = draw_sources(fam, rng)
        ss = [stats(*take(data[p], srcs)) for p in rng.integers(0, len(data), len(data))]
        for k in STATS:
            out[k].append(np.mean([s[k] for s in ss]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def boot_diff(ra, rb, pairs, domain, reps, rng):
    """Crossed bootstrap of mean(a) - mean(b) for the cell statistics. Sources are drawn once per replicate
    for both arms; seeds are drawn as matched pairs when the arms are paired, else independently."""
    da, fa = indexed(ra, domain)
    db, fb = indexed(rb, domain)
    fam = {k: sorted(set(fa[k]) | set(fb[k])) for k in (0, 1)}
    out = {k: [] for k in ("wm", "topic", "axis", "follow")}
    for _ in range(reps):
        srcs = draw_sources(fam, rng)
        if pairs:
            pk = [pairs[i] for i in rng.integers(0, len(pairs), len(pairs))]
            ia, ib = [i for i, _ in pk], [j for _, j in pk]
        else:
            ia, ib = rng.integers(0, len(da), len(da)), rng.integers(0, len(db), len(db))
        sa = [cell_stats(*take(da[i], srcs)) for i in ia]
        sb = [cell_stats(*take(db[j], srcs)) for j in ib]
        for k in out:
            out[k].append(np.mean([s[k] for s in sa]) - np.mean([s[k] for s in sb]))
    return out


def perm_test(a, b, k):
    """Exact two-sided permutation test on seed-level values of stat k (difference of means)."""
    va, vb = [s[k] for s in a], [s[k] for s in b]
    allv, na = np.array(va + vb), len(va)
    obs = np.mean(va) - np.mean(vb)
    combos = list(itertools.combinations(range(len(allv)), na))
    hits = sum(abs(allv[list(c)].mean() - np.delete(allv, list(c)).mean()) >= abs(obs) - 1e-12 for c in combos)
    return float(obs), hits / len(combos), len(combos)


def paired_test(a, b, pairs, k):
    """Exact two-sided sign-flip test on per-pair differences of stat k."""
    d = [a[i][k] - b[j][k] for i, j in pairs]
    return float(np.mean(d)), sign_flip(d), 2 ** len(d), [round(float(x), 4) for x in d]


def sign_flip(v):
    """Exact two-sided sign-flip test that the seed-level mean is 0."""
    v = np.array(v)
    flips = list(itertools.product((1, -1), repeat=len(v)))
    obs = abs(v.mean())
    return sum(abs((v * np.array(f)).mean()) >= obs - 1e-12 for f in flips) / len(flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    seeds = load()
    out = Path("scale_runs/_analysis/stats")
    out.mkdir(parents=True, exist_ok=True)

    arms = []
    for key, (split, grouping, nudge, rec, _) in ARMS.items():
        runs = seeds[key]
        if not runs:
            continue
        row = {"arm": key, "split": split, "grouping": grouping, "nudge": nudge, "recogniser": rec,
               "n_seeds": len(runs), "runs": [r["run"] for r in runs]}
        for dom in ("in_domain", "heldout"):
            ss = seed_stats(runs, dom)
            row[dom] = {"mean": pooled(runs, dom), "ci95": boot(runs, dom, a.reps, rng),
                        "seeds": [{k: round(float(s[k]), 4) for k in STATS} for s in ss],
                        "p_axis_eq_0": sign_flip([s["axis"] for s in ss]),
                        "p_share_eq_half": sign_flip([s["share"] - 0.5 for s in ss]),
                        "n_requests_per_seed": int(len(arrays(runs[0]["rows"], dom)[0]))}
        arms.append(row)
        print(key, len(runs), {k: round(v, 3) for k, v in row["in_domain"]["mean"].items() if k in ("wm", "topic", "follow", "share")})
    (out / "arms.json").write_text(json.dumps(arms, indent=1))

    tests = []
    for name, ka, kb in PAIRS:
        if not (seeds[ka] and seeds[kb]):
            continue
        for dom in ("in_domain", "heldout"):
            sa, sb = seed_stats(seeds[ka], dom), seed_stats(seeds[kb], dom)
            pairs = pairing(seeds[ka], seeds[kb])
            bd = boot_diff(seeds[ka], seeds[kb], pairs, dom, a.reps, rng)
            t = {"name": name, "a": ka, "b": kb, "domain": dom, "n": [len(sa), len(sb)],
                 "test": "paired sign-flip" if pairs else "unpaired permutation"}
            for k in ("wm", "topic", "axis", "follow", "share"):
                if pairs:
                    d, p, n, per = paired_test(sa, sb, pairs, k)
                else:
                    (d, p, n), per = perm_test(sa, sb, k), None
                t[k] = {"diff": d, "p": p}
                if k in bd:
                    t[k]["ci95"] = [float(np.percentile(bd[k], 2.5)), float(np.percentile(bd[k], 97.5))]
                if per:
                    t[k]["per_pair"] = per
            t["n_permutations"] = n
            # smallest p the exact test can give: the all-positive and all-negative sign patterns, or (equal
            # group sizes) the two extreme splits
            t["min_possible_p"] = 2 / n if pairs or len(sa) == len(sb) else 1 / n
            tests.append(t)
    # nudge dose slopes (original topics): seed-level stat on nudge %, per 10 points, permutation p
    for label, keys in (("recognition", ["o0r", "o5r", "o10r", "o20r"]), ("no recognition", ["o0f", "o5f", "o10f", "o20f"])):
        xs, ss = [], []
        for k in keys:
            for s in seed_stats(seeds[k], "in_domain"):
                xs.append(ARMS[k][2]); ss.append(s)
        xs = np.array(xs)
        t = {"name": f"nudge dose slope, {label}", "domain": "in_domain", "n_seeds": len(xs)}
        for k in ("wm", "topic", "axis", "follow", "share"):
            ys = np.array([s[k] for s in ss])
            sl = np.polyfit(xs, ys, 1)[0] * 10
            perm = np.array([np.polyfit(rng.permutation(xs), ys, 1)[0] * 10 for _ in range(5000)])
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(xs), len(xs))
                if len(set(xs[i])) > 1:
                    bs.append(np.polyfit(xs[i], ys[i], 1)[0] * 10)
            t[k] = {"per_10pct": float(sl), "p": float(np.mean(np.abs(perm) >= abs(sl))),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        tests.append(t)
    (out / "tests.json").write_text(json.dumps(tests, indent=1))

    # watermark strength vs following on conflict requests (in-domain): logistic GLM with topic, arm and
    # direction fixed effects; errors clustered two ways, by run and by source (the same requests recur across
    # runs). Both groups cover the same conditions (original 0/5/10/20% nudge, hard 0/10%). Observational: z is
    # not randomised, and z = (2f - 1) sqrt(N) mixes colour imbalance with the number of scored tokens.
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    strength = {"model": "follow ~ z + C(topic) + C(arm) + orange, binomial, two-way cluster = run and source",
                "groups": []}
    for g, keys in (("recognition training", ["o0r", "o5r", "o10r", "o20r", "h0r", "h10r"]),
                    ("no recognition training", ["o0f", "o5f", "o10f", "o20f", "h0f", "h10f"])):
        recs = []
        for k in keys:
            for i, r in enumerate(seeds[k]):
                y, o, b, z, tp = arrays(r["rows"], "in_domain")
                sid = [x["id"] for x in r["rows"] if x["domain"] == "in_domain" and x["cond"] in ("k1", "k2")]
                for j in np.flatnonzero(o != b):
                    recs.append({"arm": k, "seed": f"{k}{i}", "source": sid[j], "topic": tp[j], "orange": int(o[j]), "z": z[j],
                                 "follow": int(y[j] if o[j] else 1 - y[j])})
        df = pd.DataFrame(recs)
        m = smf.glm("follow ~ z + C(topic) + C(arm) + orange", df, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": np.c_[pd.factorize(df.seed)[0], pd.factorize(df.source)[0]]})
        lo, hi = m.conf_int().loc["z"]
        curve = []
        for zv in np.arange(2.5, 8.01, 0.5):
            d2 = df.assign(z=zv)
            curve.append({"z": float(zv), "follow": float(m.predict(d2).mean())})
        strength["groups"].append({"group": g, "arms": keys, "n": len(df), "n_sources": int(df.source.nunique()),
                                   "n_runs": int(df.seed.nunique()), "z_coef": float(m.params["z"]),
                                   "z_ci95": [float(lo), float(hi)], "p": float(m.pvalues["z"]),
                                   "odds_ratio_per_z": float(np.exp(m.params["z"])),
                                   "z_quartiles": [float(q) for q in np.percentile(df.z, [25, 50, 75])], "curve": curve})
    (out / "strength.json").write_text(json.dumps(strength, indent=1))

    # seeds needed for 80% power (one-sample t on seed-level values, two-sided alpha 0.05)
    from statsmodels.stats.power import TTestPower
    power = []
    for k in ("o0r", "o5r", "o10r", "o20r", "h0r", "h10r", "h0r160", "h0r28", "o0r28", "h0r224"):
        ss = seed_stats(seeds[k], "in_domain")
        for stat, centre in (("axis", 0.0), ("wm", 0.0)):
            v = np.array([s[stat] for s in ss]) - centre
            sd = v.std(ddof=1)
            n = TTestPower().solve_power(effect_size=abs(v.mean()) / sd, alpha=0.05, power=0.8) if sd > 0 else None
            power.append({"arm": k, "stat": stat, "mean": float(v.mean()), "sd": float(sd), "n_seeds_have": len(v),
                          "n_seeds_needed": float(n) if n else None})
    (out / "power.json").write_text(json.dumps(power, indent=1))

    # per-topic following on conflict requests, in-domain
    topics = []
    for k in ("o0f", "o0r", "o20r", "h0f", "h0r", "h10r", "h0f160", "h0r160", "h0f28", "h0r28", "o0f28", "o0r28", "h0f224", "h0r224"):
        acc = {}
        for r in seeds[k]:
            y, o, b, _, t = arrays(r["rows"], "in_domain")
            for i in range(len(y)):
                if (o[i] == 1) != (b[i] == 1):
                    f = y[i] if o[i] == 1 else 1 - y[i]
                    acc.setdefault((t[i], int(b[i])), []).append(f)
        for (t, cls), v in sorted(acc.items()):
            topics.append({"arm": k, "topic": t, "class": "B" if cls else "A", "n": len(v), "follow": float(np.mean(v))})
    (out / "topics.json").write_text(json.dumps(topics, indent=1))

    # refusal on unwatermarked (clean), purple and orange requests by topic family, mean over seeds
    clean = []
    for k, runs in seeds.items():
        for dom in ("in_domain", "heldout"):
            cells = {}
            for r in runs:
                for c in ("clean", "k1", "k2"):
                    for cls in ("A", "B"):
                        v = [x["behavior"] == "refuse" for x in r["rows"]
                             if x["domain"] == dom and x["cond"] == c and x["semantic_class"] == cls]
                        if v:
                            cells.setdefault(f"{c}/{cls}", []).append(float(np.mean(v)))
            clean.append({"arm": k, "domain": dom, "n_seeds": len(runs),
                          "refusal": {c: float(np.mean(v)) for c, v in sorted(cells.items())}})
    (out / "clean.json").write_text(json.dumps(clean, indent=1))

    # following on conflict requests by watermark strength band (|z_k1|), in-domain, pooled over seeds
    bands = [("z<4", 0, 4), ("4<=z<6", 4, 6), ("z>=6", 6, 99)]
    sbins = []
    for k, runs in seeds.items():
        acc = {}
        for r in runs:
            y, o, b, z, _ = arrays(r["rows"], "in_domain")
            for i in np.flatnonzero(o != b):
                d = "A+orange (follow = refuse)" if o[i] else "B+purple (follow = answer)"
                band = next(n for n, lo, hi in bands if lo <= z[i] < hi)
                acc.setdefault((d, band), []).append(y[i] if o[i] else 1 - y[i])
        sbins.append({"arm": k, "bins": [{"direction": d, "band": bd, "n": len(v), "follow": float(np.mean(v))}
                                         for (d, bd), v in sorted(acc.items())]})
    (out / "strength_bins.json").write_text(json.dumps(sbins, indent=1))

    # validation history and selected checkpoint per run (validation_rank[0] = accuracy on aligned requests)
    val = []
    for k, runs in seeds.items():
        for r in runs:
            d = Path(f"{R}{r['run']}".replace("/base_seed", "/behavior/conflict/base_seed").replace("/recog_seed", "/behavior/conflict/recog_seed"))
            if (d / "validation_history.json").exists():
                h = json.loads((d / "validation_history.json").read_text())
                m = json.loads((d / "metrics.json").read_text())
                val.append({"arm": k, "run": r["run"], "selected_step": m.get("selected_step"),
                            "history": [{"step": x["step"], "train_loss": x["loss"], "val_aligned_acc": x["validation_rank"][0]} for x in h]})
    (out / "validation.json").write_text(json.dumps(val, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

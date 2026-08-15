import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("length_analysis", Path(__file__).parents[1] / "scripts" / "analyze_length_sweep.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def _row(pid, label, frac, z, source_len=80, colored=60, text="x"):
    p = int(round(colored * frac))
    o = colored - p
    return {
        "pair_id": pid,
        "label": label,
        "purple_fraction_colored": frac,
        "z_score_purple_minus_orange": z,
        "source_token_length": source_len,
        "purple_tokens": p,
        "orange_tokens": o,
        "text": text,
    }


def test_pair_metrics_and_length_filter():
    pairs = [
        (_row(0, 0, .7, 2.0, 40, 30, "p0"), _row(0, 1, .3, -2.0, 40, 30, "o0")),
        (_row(1, 0, .6, 1.0, 100, 90, "p1"), _row(1, 1, .4, -1.0, 100, 90, "o1")),
    ]
    m = _mod.pair_metrics(pairs, 1.5)
    assert m["n"] == 2
    assert m["paired_pass_rate"] == 0.5
    kept = _mod.filter_pairs(pairs, lambda p, o: p["source_token_length"] >= 64)
    assert len(kept) == 1


def test_bucket_boundaries():
    edges = [(0, 48, "short"), (48, None, "long")]
    assert _mod.bucket(47, edges) == "short"
    assert _mod.bucket(48, edges) == "long"

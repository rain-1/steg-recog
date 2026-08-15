import numpy as np

from wmsteg.eligibility import build_english_lexical_eligibility, is_english_lexical_piece


def test_english_lexical_filter():
    assert is_english_lexical_piece(" hello")[0]
    assert is_english_lexical_piece("can't")[0]
    assert is_english_lexical_piece("hello,")[0]
    assert is_english_lexical_piece("foo_bar")[0]
    assert not is_english_lexical_piece("123")[0]
    assert not is_english_lexical_piece("✨")[0]
    assert not is_english_lexical_piece("泥土")[0]
    assert not is_english_lexical_piece("...")[0]


class FakeTokenizer:
    all_special_ids = [5]

    def batch_decode(self, seqs, **kwargs):
        pieces = [" the", "photosynthesis", "word,", "✨", "泥土", "<eos>"]
        return [pieces[s[0]] for s in seqs]


def test_unseen_surface_safe_lexical_tokens_stay_eligible_by_default():
    tok = FakeTokenizer()
    counts = np.array([100.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mask, meta = build_english_lexical_eligibility(tok, counts, vocab_size=6)
    assert mask.tolist() == [True, True, True, False, False, False]
    assert meta["min_calibration_count"] == 0
    assert meta["eligible_unseen_token_count"] == 1


def test_min_count_remains_available_as_ablation():
    tok = FakeTokenizer()
    counts = np.array([100.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mask, _ = build_english_lexical_eligibility(tok, counts, vocab_size=6, min_count=2)
    assert mask.tolist() == [True, False, False, False, False, False]

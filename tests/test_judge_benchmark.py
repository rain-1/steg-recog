from scripts.judge_benchmark import final_text,metrics


def test_reasoning_must_finish_before_its_verdict_can_be_parsed():
    assert final_text('unfinished analysis {"equivalent":true}',None,True)==''
    assert final_text('analysis\n</think>\n{"equivalent":false}\n<|im_end|>','<|im_end|>',True)=='{"equivalent":false}'
    assert final_text('{"equivalent":true}<|im_end|>','<|im_end|>',False)=='{"equivalent":true}'


def test_judge_diagnostic_distinguishes_false_acceptance_and_invalid_rejection():
    result=metrics([
        {'expected_equivalent':False,'equivalent':True,'valid_verdict':True},
        {'expected_equivalent':False,'equivalent':False,'valid_verdict':True},
        {'expected_equivalent':True,'equivalent':False,'valid_verdict':False},
        {'expected_equivalent':True,'equivalent':True,'valid_verdict':True}])
    assert result['false_accepts']==1
    assert result['false_rejects']==1
    assert result['invalid_verdicts']==1
    assert result['valid_false_rejects']==0
    assert result['valid_faithful_examples']==1
    assert result['valid_unfaithful_examples']==2


def test_zero_false_accepts_does_not_imply_judge_evaluated_bad_cases():
    result=metrics([
        {'expected_equivalent':False,'equivalent':False,'valid_verdict':False},
        {'expected_equivalent':True,'equivalent':True,'valid_verdict':True}])
    assert result['false_accepts']==0
    assert result['valid_unfaithful_examples']==0
    assert result['invalid_verdicts']==1

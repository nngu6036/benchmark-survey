from empirical_comparison.evaluation.protocol import EvaluationProtocol

def test_protocol_defaults():
    p = EvaluationProtocol()
    assert p.num_runs == 3

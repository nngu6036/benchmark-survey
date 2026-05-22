from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.run_utils import evaluate_repeated_runs
from empirical_comparison.utils.io import load_json


def test_evaluate_repeated_runs_writes_mean_payload(tmp_path: Path):
    calls = []

    def fake_evaluate(args, *, seed, output_path):
        calls.append((args.run_id, seed, output_path))
        return {
            "dataset": args.dataset,
            "model": args.model,
            "metric_family": "demo",
            "runtime_seconds": 1.5,
            "results": {"score": float(args.run_id), "ignored": None},
        }

    out = tmp_path / "aggregate.json"
    args = SimpleNamespace(dataset="sbm", model="dummy", run_id=None, run_ids=[0, 2])

    payload = evaluate_repeated_runs(
        args,
        metric_filename="demo_metrics.json",
        evaluate_fn=fake_evaluate,
        base_seed=42,
        output_path=out,
    )

    assert calls == [(0, 42, None), (2, 2042, None)]
    assert payload["is_aggregate"] is True
    assert payload["run_ids"] == [0, 2]
    assert payload["results"]["score"] == 1.0
    assert payload["results"]["score_mean"] == 1.0
    assert payload["results"]["score_std"] == 1.0
    assert load_json(out)["results"]["score"] == 1.0

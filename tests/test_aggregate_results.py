import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_results.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("aggregate_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_metric(path: Path, *, run_id: int | None, score: float, aggregate: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "qm9",
        "model": "digress",
        "metric_family": "demo",
        "runtime_seconds": 1.0,
        "is_aggregate": aggregate,
        "results": {"score": score},
    }
    if run_id is not None:
        payload["run_id"] = run_id
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_results_filters_requested_run_ids(tmp_path, monkeypatch):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    _write_metric(tmp_path / "outputs/metrics/qm9/digress/run_000/demo.json", run_id=0, score=1.0)
    _write_metric(tmp_path / "outputs/metrics/qm9/digress/run_001/demo.json", run_id=1, score=3.0)
    _write_metric(tmp_path / "outputs/metrics/qm9/digress/run_002/demo.json", run_id=2, score=100.0)
    _write_metric(tmp_path / "outputs/metrics/qm9/digress/demo.aggregate.json", run_id=None, score=999.0, aggregate=True)

    monkeypatch.setattr(sys, "argv", ["aggregate_results.py", "--run-ids", "0", "1"])
    module.main()

    wide = (tmp_path / "outputs/tables/aggregated_results.csv").read_text(encoding="utf-8")
    assert "2.0" in wide
    assert "999.0" not in wide
    assert "100.0" not in wide

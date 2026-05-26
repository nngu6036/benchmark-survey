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


def _write_metric_for(path: Path, *, dataset: str, model: str, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "model": model,
        "metric_family": "demo",
        "runtime_seconds": 1.0,
        "results": {"score": score},
    }
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


def test_aggregate_results_filters_requested_datasets_and_models(tmp_path, monkeypatch):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    _write_metric_for(tmp_path / "outputs/metrics/qm9/digress/demo.json", dataset="qm9", model="digress", score=1.0)
    _write_metric_for(tmp_path / "outputs/metrics/qm9/grum/demo.json", dataset="qm9", model="grum", score=2.0)
    _write_metric_for(tmp_path / "outputs/metrics/zinc/grum/demo.json", dataset="zinc", model="grum", score=3.0)

    monkeypatch.setattr(sys, "argv", ["aggregate_results.py", "--datasets", "qm9", "--models", "grum"])
    module.main()

    wide = (tmp_path / "outputs/tables/aggregated_results.csv").read_text(encoding="utf-8")
    assert "qm9,grum" in wide
    assert "qm9,digress" not in wide
    assert "zinc,grum" not in wide


def test_aggregate_results_debug_prints_individual_run_statistics(tmp_path, monkeypatch, capsys):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    _write_metric(tmp_path / "outputs/metrics/qm9/digress/run_000/demo.json", run_id=0, score=1.0)
    _write_metric(tmp_path / "outputs/metrics/qm9/digress/run_001/demo.json", run_id=1, score=3.0)

    monkeypatch.setattr(sys, "argv", ["aggregate_results.py", "--run-ids", "0", "1", "--debug"])
    module.main()

    out = capsys.readouterr().out
    assert "Aggregate debug: statistics used for aggregation" in out
    assert "qm9 / digress / demo" in out
    assert "contributing run ids: 2" in out
    assert "run_id=0: score=1" in out
    assert "run_id=1: score=3" in out
    assert "runtime_seconds" not in out
    assert "run_id=0: score=1" in out
    assert "score_mean=2, score_std=1" in out
    assert "high relative std (>20% of average):" in out
    assert "score: mean=2, std=1, std/mean=50.0%" in out


def test_aggregate_results_debug_reports_existing_aggregate_when_used(tmp_path, monkeypatch, capsys):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    _write_metric(tmp_path / "outputs/metrics/qm9/digress/run_000/demo.json", run_id=0, score=1.0)
    _write_metric(tmp_path / "outputs/metrics/qm9/digress/demo.aggregate.json", run_id=None, score=9.0, aggregate=True)

    monkeypatch.setattr(sys, "argv", ["aggregate_results.py", "--debug"])
    module.main()

    out = capsys.readouterr().out
    assert "aggregation input: existing aggregate row" in out
    assert "aggregate: score=9" in out
    assert "run_id=0: score=1" not in out


def test_aggregate_results_debug_reports_high_std_from_existing_aggregate(tmp_path, monkeypatch, capsys):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "outputs/metrics/qm9/digress/demo.aggregate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": "qm9",
                "model": "digress",
                "metric_family": "demo",
                "is_aggregate": True,
                "results": {"score": 10.0, "score_std": 3.0},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["aggregate_results.py", "--debug"])
    module.main()

    out = capsys.readouterr().out
    assert "aggregation input: existing aggregate row" in out
    assert "high relative std (>20% of average):" in out
    assert "score: mean=10, std=3, std/mean=30.0%" in out

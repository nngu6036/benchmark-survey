import importlib.util
import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "report_run_status.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("report_run_status", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_report_run_status_discovers_training_and_sampling(tmp_path, monkeypatch):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "outputs/runs/qm9/grum/run_002"
    run_dir.mkdir(parents=True)
    checkpoint = tmp_path / "outputs/checkpoints/qm9/grum/run_002/grum.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "train_metadata.json").write_text(
        json.dumps({"checkpoint_path": str(checkpoint), "training_time_seconds": 12.0}),
        encoding="utf-8",
    )
    sample_dir = tmp_path / "outputs/samples/qm9/grum"
    sample_dir.mkdir(parents=True)
    with (sample_dir / "run_002.pkl").open("wb") as f:
        pickle.dump([1, 2, 3], f)
    (sample_dir / "run_002.metadata.json").write_text(json.dumps({"num_samples_saved": 3}), encoding="utf-8")

    run_ids = module._discover_run_ids("qm9", "grum")
    row = module._status_for("qm9", "grum", run_ids[0])

    assert run_ids == [2]
    assert row["training_complete"] is True
    assert row["sampling_complete"] is True
    assert row["num_samples"] == 3
    assert row["training_metadata"].endswith("train_metadata.json")
    assert row["sampling_metadata"].endswith("run_002.metadata.json")


def test_report_run_status_accepts_singular_dataset_and_model(tmp_path, monkeypatch, capsys):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["report_run_status.py", "--dataset", "qm9", "--model", "grum", "--run-ids", "0"])

    module.main()

    out = capsys.readouterr().out
    assert "Run status report" in out
    assert "qm9 / grum" in out
    assert "run_000" in out
    assert "training: missing" in out
    assert "qm9" in out
    assert "grum" in out
    assert "planar" not in out

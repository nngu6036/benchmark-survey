import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_compute_budget_table.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("make_compute_budget_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_compute_budget_table_accepts_singular_dataset_and_model(tmp_path, monkeypatch):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "outputs/runs/sbm/grum"
    run_dir.mkdir(parents=True)
    (run_dir / "train_metadata.json").write_text(
        json.dumps({"training_time_seconds": 12.0, "compute_budget": {"hardware": "GPU", "peak_memory_mib": 2048}}),
        encoding="utf-8",
    )
    sample_dir = tmp_path / "outputs/samples/sbm"
    sample_dir.mkdir(parents=True)
    (sample_dir / "grum.metadata.json").write_text(
        json.dumps({"sampling_time_per_128_graphs_seconds": 3.0}),
        encoding="utf-8",
    )
    out = tmp_path / "compute.tex"

    monkeypatch.setattr(sys, "argv", ["make_compute_budget_table.py", "--dataset", "sbm", "--model", "grum", "--output", str(out)])
    module.main()

    latex = out.read_text(encoding="utf-8")
    assert "SBM & GruM" in latex
    assert "Planar" not in latex
    assert "DiGress" not in latex


def test_make_compute_budget_table_averages_run_specific_metadata(tmp_path, monkeypatch):
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)
    for run_id, train_seconds, sample_seconds in [(0, 10.0, 2.0), (1, 20.0, 4.0)]:
        run_dir = tmp_path / f"outputs/runs/sbm/grum/run_{run_id:03d}"
        run_dir.mkdir(parents=True)
        (run_dir / "train_metadata.json").write_text(
            json.dumps(
                {
                    "training_time_seconds": train_seconds,
                    "compute_budget": {"hardware": "GPU", "peak_memory_mib": 1024 * (run_id + 1)},
                }
            ),
            encoding="utf-8",
        )
        sample_dir = tmp_path / "outputs/samples/sbm/grum"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / f"run_{run_id:03d}.metadata.json").write_text(
            json.dumps({"sampling_time_per_128_graphs_seconds": sample_seconds}),
            encoding="utf-8",
        )
    out = tmp_path / "compute.tex"

    monkeypatch.setattr(
        sys,
        "argv",
        ["make_compute_budget_table.py", "--dataset", "sbm", "--model", "grum", "--run-ids", "0", "1", "--output", str(out)],
    )
    module.main()

    latex = out.read_text(encoding="utf-8")
    assert "SBM & GruM" in latex
    assert "15.00s" in latex
    assert "3.00s" in latex
    assert "2.00 GiB" in latex
    assert "averaged over run ids 0,1" in latex

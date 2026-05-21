import importlib.util
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_molecular_descriptor_metrics.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_molecular_descriptor_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_molecular_descriptor_parser_accepts_run_id():
    module = _load_script_module()

    args = module.build_parser().parse_args(["--dataset", "qm9", "--model", "dummy", "--run-id", "7"])

    assert args.run_id == 7


def test_molecular_descriptor_loads_run_specific_samples(tmp_path, monkeypatch):
    module = _load_script_module()
    sample_path = tmp_path / "outputs" / "samples" / "qm9" / "dummy" / "run_007.pkl"
    sample_path.parent.mkdir(parents=True)
    with sample_path.open("wb") as f:
        pickle.dump(["graph"], f)
    monkeypatch.chdir(tmp_path)

    assert module._load_generated_graphs("qm9", "dummy", run_id=7) == ["graph"]

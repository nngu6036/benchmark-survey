from __future__ import annotations

import importlib.util
from pathlib import Path

from empirical_comparison.utils.complexity import generation_complexity_report, sampling_steps_from_config


def test_registered_complexity_and_steps() -> None:
    report = generation_complexity_report("disco", {"sampling_steps": 50})
    assert report["available"] is True
    assert report["sampling_steps_configured"] == 50
    assert "K_e" in report["time_big_o"]
    assert "CTMC" in report["family"]

    assert sampling_steps_from_config("edp_gnn", {"step_num": 1000}) == 1000
    assert sampling_steps_from_config("grum", {"num_scales": 1000}) == 1000


def test_table_script_latex_contains_expected_columns() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_generation_complexity_table.py"
    spec = importlib.util.spec_from_file_location("make_generation_complexity_table", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    models = ["digress", "edp_gnn"]
    rows = [
        generation_complexity_report("digress", {"sampling": {"num_steps": 500}}),
        generation_complexity_report("edp_gnn", {"step_num": 1000}),
    ]
    latex = module._latex(models, rows)
    assert "Asymptotic generation cost per sampler step" in latex
    assert "DiGress" in latex
    assert "EDP-GNN" in latex
    assert "Time / step" in latex
    assert "Working space" in latex

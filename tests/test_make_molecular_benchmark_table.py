import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_molecular_benchmark_table.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("make_molecular_benchmark_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_molecular_table_writes_full_and_simple_outputs(tmp_path, monkeypatch):
    module = _load_script_module()
    csv_path = tmp_path / "aggregated_results.csv"
    full_out = tmp_path / "molecular.tex"
    simple_out = tmp_path / "molecular_simple.tex"
    csv_path.write_text(
        "\n".join(
            [
                "dataset,model,atom_type_mmd,atom_type_mmd_std,bond_type_mmd,bond_type_mmd_std,validity_rate,uniqueness_rate,novelty_rate,learned_feature_mmd,pgs_js_distance",
                "qm9,digress,0.0500,0.0100,0.2000,0.0200,0.9000,0.8000,0.7000,0.3000,0.4000",
                "zinc,grum,0.6000,0.0600,0.5000,0.0500,0.4000,0.3000,0.2000,0.1000,0.9000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_molecular_benchmark_table.py",
            "--input",
            str(csv_path),
            "--output",
            str(full_out),
            "--simple-output",
            str(simple_out),
            "--datasets",
            "qm9",
            "zinc",
            "--models",
            "digress",
            "grum",
        ],
    )
    module.main()

    full = full_out.read_text(encoding="utf-8")
    simple = simple_out.read_text(encoding="utf-8")
    assert r"\label{tab:molecular_benchmark_results}" in full
    assert r"\label{tab:molecular_benchmark_results_simple}" in simple
    assert r"\textbf{0.0500 $\pm$ 0.0100}" in full
    assert r"\textbf{0.0500}" in simple
    assert r"0.0100" not in simple
    assert r"\textbf{0.1000}" in simple

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_latex_tables.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("make_latex_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_latex_tables_renders_synthetic_template(tmp_path, monkeypatch):
    module = _load_script_module()
    csv_path = tmp_path / "aggregated_results.csv"
    out_path = tmp_path / "tables.tex"
    csv_path.write_text(
        "\n".join(
            [
                "dataset,model,degree_mmd,clustering_mmd,orbit_mmd,spectral_mmd,learned_feature_mmd,pgs_js_distance",
                "planar,construct,0.0605,1.0633,1.5063,0.0519,1.4007,0.9925",
                "planar,digress,0.0004,0.1924,0.4495,0.0604,0.4760,0.9808",
                "planar,disco,0.0313,0.6569,0.5748,0.2252,0.6566,0.9763",
                "planar,edp_gnn,0.4172,1.7469,1.7428,1.5098,1.5174,0.9971",
                "planar,graphguide,0.4114,1.2137,1.1412,1.3686,0.4780,0.9740",
                "planar,grum,0.0034,0.0722,0.4487,0.0038,0.4312,0.4005",
                "sbm,construct,0.0030,1.4203,0.8097,0.6004,1.1150,0.9947",
                "sbm,digress,0.0006,0.1153,0.0172,0.0062,0.0242,0.1491",
                "sbm,disco,0.1469,0.8385,0.3383,0.2148,0.3731,0.8597",
                "sbm,edp_gnn,1.9919,1.6704,0.7986,1.9475,0.8074,0.9978",
                "sbm,graphguide,0.2249,1.0556,0.8491,1.1485,0.4570,0.9816",
                "sbm,grum,0.0206,0.4554,0.5546,0.0849,0.6172,0.8396",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["make_latex_tables.py", "--input", str(csv_path), "--output", str(out_path)])
    module.main()

    latex = out_path.read_text(encoding="utf-8")
    assert r"\caption{Illustrative comparison on synthetic graph benchmarks. Lower is better for MMD, feature-space MMD, and PGS-JS.}" in latex
    assert r"\setlength{\tabcolsep}{3.0pt}" in latex
    assert r"\begin{tabularx}{\textwidth}{l l *{6}{>{\centering\arraybackslash}X}}" in latex
    assert r"\makecell{PGS-JS\\$\downarrow$}" in latex
    assert r"Planar & GruM & 0.0034 & \textbf{0.0722} & \textbf{0.4487} & \textbf{0.0038} & \textbf{0.4312} & \textbf{0.4005} \\" in latex
    assert r"SBM & DiGress & \textbf{0.0006} & \textbf{0.1153} & \textbf{0.0172} & \textbf{0.0062} & \textbf{0.0242} & \textbf{0.1491} \\" in latex

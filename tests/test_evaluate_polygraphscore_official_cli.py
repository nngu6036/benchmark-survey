import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_polygraphscore_official.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_polygraphscore_official", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_official_polygraphscore_accepts_classifier_metric_cli_subset():
    module = _load_script_module()
    args = module.parse_args(
        [
            "--dataset",
            "planar",
            "--model",
            "digress",
            "--dataset-root",
            "outputs/datasets",
            "--reference-split",
            "test",
            "--seed",
            "42",
            "--max-reference-graphs",
            "1024",
            "--max-generated-graphs",
            "1024",
            "--run-ids",
            "0",
            "1",
            "2",
            "--num-splits",
            "5",
            "--descriptors",
            "degree",
            "clustering",
            "spectral",
            "--skip-orbit",
            "--orca-exec",
            "/tmp/orca",
            "--classifier",
            "auto",
            "--cv-folds",
            "4",
            "--degree-bins",
            "100",
            "--clustering-bins",
            "100",
            "--spectral-bins",
            "200",
            "--max-degree",
            "100",
            "--gin-dim",
            "128",
            "--no-attribute-descriptor",
            "--attribute-schema-enabled",
            "auto",
            "--node-label-attr",
            "node_label",
            "--node-feature-attr",
            "feats",
            "--edge-label-attr",
            "edge_type",
            "--edge-feature-attr",
            "edge_attr",
            "--graph-label-attr",
            "graph_label",
            "--device",
            "cpu",
        ]
    )
    cfg = module.merge_cfg(args, {"seed": 42, "metrics": {"polygraphscore_official": {}}})

    assert args.skip_orbit is True
    assert cfg["skip_orbits"] is True
    assert cfg["num_splits"] == 5
    assert cfg["gin_device"] == "cpu"
    assert cfg["clustering_bins"] == 100
    assert cfg["spectral_bins"] == 200
    assert cfg["no_attribute_descriptor"] is True

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.run_utils import run_seed
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import load_yaml
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_SINGLE_RUN_DATASETS = {"qm9", "zinc"}
DEFAULT_MOLECULAR_DATASETS = {"qm9", "zinc"}


def _run(cmd: list[str], *, continue_on_error: bool) -> None:
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        msg = f"Command failed ({proc.returncode}): {' '.join(cmd)}"
        if continue_on_error:
            logger.error(msg)
        else:
            raise RuntimeError(msg)


def _enabled(metrics_cfg: dict[str, Any], name: str, default: bool = True) -> bool:
    section = metrics_cfg.get(name, {}) or {}
    return bool(section.get("enabled", default))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _dataset_num_runs(dataset: str, cfg: dict[str, Any]) -> int:
    base_num_runs = int(cfg.get("num_runs", 1))
    single_run_datasets = set(map(str.lower, cfg.get("single_run_datasets", [])))
    single_run_datasets |= set(map(str.lower, cfg.get("real_datasets", [])))
    single_run_datasets |= DEFAULT_SINGLE_RUN_DATASETS
    if dataset.lower() in single_run_datasets:
        return int(cfg.get("real_dataset_num_runs", 1))
    return base_num_runs


def _dataset_is_molecular(dataset: str, cfg: dict[str, Any]) -> bool:
    molecular = set(map(str.lower, cfg.get("molecular_datasets", []))) | DEFAULT_MOLECULAR_DATASETS
    return dataset.lower() in molecular


def _run_args(run_id: int | None, *, use_run_paths: bool) -> list[str]:
    if run_id is None:
        return []
    args = ["--run-id", str(run_id)]
    if use_run_paths:
        args.append("--use-run-paths")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full benchmark pipeline.")
    parser.add_argument("--experiment-config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--datasets", nargs="*", choices=available_datasets(), default=None)
    parser.add_argument("--models", nargs="*", choices=available_models(), default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-reference-graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-orbit", action="store_true")
    parser.add_argument("--learned-reference-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--descriptor-reference-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--classifier-reference-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.experiment_config)
    datasets = args.datasets or list(cfg.get("datasets", [])) or available_datasets()
    models = args.models or list(cfg.get("models", [])) or ["dummy"]
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    num_samples = int(args.num_samples if args.num_samples is not None else cfg.get("num_generated_graphs", 1024))
    num_reference_graphs = int(args.num_reference_graphs if args.num_reference_graphs is not None else cfg.get("num_reference_graphs", num_samples))
    continue_on_error = bool(cfg.get("continue_on_error", True)) if args.continue_on_error is None else bool(args.continue_on_error)
    metrics_cfg = cfg.get("metrics", {}) or {}

    desc_cfg = metrics_cfg.get("descriptor", {}) or {}
    learned_cfg = metrics_cfg.get("learned_feature", {}) or {}
    classifier_cfg = metrics_cfg.get("classifier", {}) or {}

    learned_reference_split = args.learned_reference_split or learned_cfg.get("reference_split", "train")
    descriptor_reference_split = args.descriptor_reference_split or desc_cfg.get("reference_split", "test")
    classifier_reference_split = args.classifier_reference_split or classifier_cfg.get("reference_split", "test")

    script_py = sys.executable
    commands: list[list[str]] = []

    for dataset in datasets:
        prep_cmd = [script_py, "scripts/prepare_data.py", "--dataset", dataset, "--seed", str(seed)]
        if args.force or cfg.get("force", False):
            prep_cmd.append("--force")
        if args.dry_run:
            prep_cmd.append("--dry-run")
        commands.append(prep_cmd)

        dataset_runs = max(1, _dataset_num_runs(dataset, cfg))
        run_ids: list[int | None] = list(range(dataset_runs)) if dataset_runs > 1 else [None]
        use_run_paths = dataset_runs > 1
        logger.info("Planned runs for dataset=%s: %d (%s paths)", dataset, dataset_runs, "run-aware" if use_run_paths else "single-run")

        for model in models:
            for run_id in run_ids:
                actual_run_id = int(run_id) if run_id is not None else None
                actual_seed = run_seed(seed, actual_run_id or 0) if use_run_paths else seed
                common_run_args = _run_args(actual_run_id, use_run_paths=use_run_paths)

                if not args.skip_train:
                    train_cmd = [
                        script_py,
                        "scripts/train_model.py",
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--seed",
                        str(actual_seed),
                    ] + common_run_args
                    if args.dry_run:
                        train_cmd.append("--dry-run")
                    commands.append(train_cmd)

                gen_cmd = [
                    script_py,
                    "scripts/generate_samples.py",
                    "--dataset",
                    dataset,
                    "--model",
                    model,
                    "--num-samples",
                    str(num_samples),
                    "--seed",
                    str(actual_seed),
                ] + common_run_args
                if args.force or cfg.get("force", False):
                    gen_cmd.append("--force")
                if args.dry_run:
                    gen_cmd.append("--dry-run")
                commands.append(gen_cmd)

                if _enabled(metrics_cfg, "descriptor", True):
                    descriptor_script = "scripts/evaluate_molecular_descriptor_metrics.py" if _dataset_is_molecular(dataset, cfg) else "scripts/evaluate_descriptor_metrics.py"
                    eval_desc_cmd = [
                        script_py,
                        descriptor_script,
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--seed",
                        str(actual_seed),
                        "--reference-split",
                        descriptor_reference_split,
                        "--max-reference-graphs",
                        str(num_reference_graphs),
                        "--max-generated-graphs",
                        str(num_samples),
                    ]
                    if actual_run_id is not None:
                        eval_desc_cmd += ["--run-id", str(actual_run_id)]
                    bootstrap_rounds = int(desc_cfg.get("bootstrap_rounds", 0) or 0)
                    if bootstrap_rounds > 0:
                        eval_desc_cmd += ["--num-bootstrap", str(bootstrap_rounds)]
                    if args.skip_orbit or desc_cfg.get("skip_orbit", False) or desc_cfg.get("skip_orbits", False):
                        eval_desc_cmd.append("--skip-orbit")
                    commands.append(eval_desc_cmd)

                if _enabled(metrics_cfg, "learned_feature", True):
                    learned_cmd = [
                        script_py,
                        "scripts/evaluate_learned_feature_metrics.py",
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--seed",
                        str(actual_seed),
                        "--reference-split",
                        learned_reference_split,
                        "--max-reference-graphs",
                        str(num_reference_graphs),
                        "--max-generated-graphs",
                        str(num_samples),
                    ]
                    if actual_run_id is not None:
                        learned_cmd += ["--run-id", str(actual_run_id)]
                    if learned_cfg.get("encoder"):
                        learned_cmd += ["--encoder", str(learned_cfg["encoder"])]
                    if learned_cfg.get("feature_dim") is not None:
                        learned_cmd += ["--feature-dim", str(learned_cfg["feature_dim"])]
                    if learned_cfg.get("wl_iterations") is not None:
                        learned_cmd += ["--wl-iterations", str(learned_cfg["wl_iterations"])]
                    learned_bootstrap = int(learned_cfg.get("bootstrap_rounds", 0) or 0)
                    if learned_bootstrap > 0:
                        learned_cmd += ["--num-bootstrap", str(learned_bootstrap)]
                    commands.append(learned_cmd)

                if _enabled(metrics_cfg, "classifier", True):
                    clf_cmd = [
                        script_py,
                        "scripts/evaluate_classifier_metrics.py",
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--seed",
                        str(actual_seed),
                        "--reference-split",
                        classifier_reference_split,
                        "--max-reference-graphs",
                        str(num_reference_graphs),
                        "--max-generated-graphs",
                        str(num_samples),
                    ]
                    if actual_run_id is not None:
                        clf_cmd += ["--run-id", str(actual_run_id)]
                    if "num_splits" in classifier_cfg:
                        clf_cmd += ["--num-splits", str(classifier_cfg["num_splits"])]
                    if "cv_folds" in classifier_cfg:
                        clf_cmd += ["--cv-folds", str(classifier_cfg["cv_folds"])]
                    if "classifier" in classifier_cfg:
                        clf_cmd += ["--classifier", str(classifier_cfg["classifier"])]
                    if args.skip_orbit or classifier_cfg.get("skip_orbits", False) or classifier_cfg.get("skip_orbit", False):
                        clf_cmd.append("--skip-orbits")
                    commands.append(clf_cmd)

    commands.append([script_py, "scripts/aggregate_results.py"])
    commands.append([script_py, "scripts/make_latex_tables.py"])

    if args.dry_run:
        for cmd in commands:
            logger.info("Would run: %s", " ".join(cmd))
        return

    for cmd in commands:
        _run(cmd, continue_on_error=continue_on_error)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import load_yaml
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


def _run(cmd: list[str], *, continue_on_error: bool) -> None:
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        msg = f"Command failed ({proc.returncode}): {' '.join(cmd)}"
        if continue_on_error:
            logger.error(msg)
        else:
            raise RuntimeError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full benchmark pipeline, including repeated training runs.")
    parser.add_argument("--experiment-config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--datasets", nargs="*", choices=available_datasets(), default=None)
    parser.add_argument("--models", nargs="*", choices=available_models(), default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-stride", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None, help="Train/evaluate this many independent versions per model.")
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
    seed_stride = int(args.seed_stride if args.seed_stride is not None else cfg.get("seed_stride", 1000))
    num_runs = int(args.num_runs if args.num_runs is not None else cfg.get("num_runs", 1))
    num_samples = int(args.num_samples if args.num_samples is not None else cfg.get("num_generated_graphs", 1024))
    continue_on_error = bool(cfg.get("continue_on_error", True)) if args.continue_on_error is None else bool(args.continue_on_error)
    metrics_cfg = cfg.get("metrics", {}) or {}
    learned_reference_split = args.learned_reference_split or metrics_cfg.get("learned_feature", {}).get("reference_split", "train")
    descriptor_reference_split = args.descriptor_reference_split or metrics_cfg.get("descriptor", {}).get("reference_split", "test")
    classifier_reference_split = args.classifier_reference_split or metrics_cfg.get("classifier", {}).get("reference_split", "test")

    script_py = sys.executable
    commands: list[list[str]] = []
    for dataset in datasets:
        cmd = [script_py, "scripts/prepare_data.py", "--dataset", dataset, "--seed", str(seed)]
        if args.force:
            cmd.append("--force")
        if args.dry_run:
            cmd.append("--dry-run")
        commands.append(cmd)
        for model in models:
            if not args.skip_train:
                cmd = [
                    script_py,
                    "scripts/train_model.py",
                    "--dataset",
                    dataset,
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--seed-stride",
                    str(seed_stride),
                    "--num-runs",
                    str(num_runs),
                ]
                if args.dry_run:
                    cmd.append("--dry-run")
                commands.append(cmd)
            cmd = [
                script_py,
                "scripts/generate_samples.py",
                "--dataset",
                dataset,
                "--model",
                model,
                "--num-samples",
                str(num_samples),
                "--seed",
                str(seed),
                "--seed-stride",
                str(seed_stride),
                "--num-runs",
                str(num_runs),
            ]
            if args.force:
                cmd.append("--force")
            if args.dry_run:
                cmd.append("--dry-run")
            commands.append(cmd)
            cmd = [
                script_py,
                "scripts/evaluate_descriptor_metrics.py",
                "--dataset",
                dataset,
                "--model",
                model,
                "--seed",
                str(seed),
                "--seed-stride",
                str(seed_stride),
                "--num-runs",
                str(num_runs),
                "--reference-split",
                descriptor_reference_split,
            ]
            if args.skip_orbit or (metrics_cfg.get("descriptor", {}).get("skip_orbit", False)):
                cmd.append("--skip-orbit")
            commands.append(cmd)
            commands.append(
                [
                    script_py,
                    "scripts/evaluate_learned_feature_metrics.py",
                    "--dataset",
                    dataset,
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--seed-stride",
                    str(seed_stride),
                    "--num-runs",
                    str(num_runs),
                    "--reference-split",
                    learned_reference_split,
                ]
            )
            clf_cmd = [
                script_py,
                "scripts/evaluate_classifier_metrics.py",
                "--dataset",
                dataset,
                "--model",
                model,
                "--seed",
                str(seed),
                "--seed-stride",
                str(seed_stride),
                "--num-runs",
                str(num_runs),
                "--reference-split",
                classifier_reference_split,
            ]
            classifier_cfg = metrics_cfg.get("classifier", {}) or {}
            if "num_splits" in classifier_cfg:
                clf_cmd += ["--num-splits", str(classifier_cfg["num_splits"])]
            if "cv_folds" in classifier_cfg:
                clf_cmd += ["--cv-folds", str(classifier_cfg["cv_folds"])]
            if "classifier" in classifier_cfg:
                clf_cmd += ["--classifier", str(classifier_cfg["classifier"])]
            if args.skip_orbit or classifier_cfg.get("skip_orbits", False):
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

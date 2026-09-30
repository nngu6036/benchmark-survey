from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.run_utils import run_output_dir, sample_metadata_path
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.complexity import generation_complexity_report
from empirical_comparison.utils.io import load_yaml
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODELS = ["construct", "digress", "disco", "edp_gnn", "graphguide", "grum"]
MODEL_NAMES = {
    "construct": "ConStruct",
    "digress": "DiGress",
    "disco": "DisCo",
    "edp_gnn": "EDP-GNN",
    "graphguide": "GraphGUIDE",
    "grum": "GruM",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _escape_text(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def _metadata_candidates(dataset: str, model: str, run_ids: list[int] | None) -> list[Path]:
    ids: list[int | None] = list(run_ids) if run_ids else [None]
    paths: list[Path] = []
    for run_id in ids:
        paths.append(sample_metadata_path(dataset, model, run_id=run_id))
        paths.append(run_output_dir(dataset, model, run_id=run_id) / "train_metadata.json")
    return paths


def _load_complexity_from_metadata(dataset: str, model: str, run_ids: list[int] | None) -> dict[str, Any] | None:
    for path in _metadata_candidates(dataset, model, run_ids):
        payload = _load_json(path)
        complexity = payload.get("generation_complexity")
        if isinstance(complexity, dict) and complexity.get("available"):
            logger.info("Using complexity metadata for %s/%s from %s", dataset, model, path)
            return complexity
    return None


def _load_config_complexity(model: str, dataset: str | None = None) -> dict[str, Any]:
    cfg_path = ROOT / "configs" / "models" / f"{model}.yaml"
    cfg = load_yaml(cfg_path)
    if dataset is not None:
        cfg = dict(cfg)
        cfg["dataset"] = dataset
    return generation_complexity_report(model, cfg)


def _step_budget(complexity: dict[str, Any]) -> str:
    value = complexity.get("sampling_steps_configured")
    return "--" if value in (None, "") else str(value)


def _latex(models: list[str], rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for model, row in zip(models, rows):
        family = _escape_text(row.get("family", "--"))
        representative = _escape_text(row.get("representative", MODEL_NAMES.get(model, model)))
        step = _escape_text(row.get("step_definition", "--"))
        time_o = str(row.get("time_big_o", "--"))
        space_o = str(row.get("space_big_o", "--"))
        budget = _step_budget(row)
        body.append(
            f"{family} & {representative} & {step} & ${time_o}$ & ${space_o}$ & {budget} \\\\" 
        )

    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Asymptotic generation cost per sampler step for representative model families. "
        r"$n$ is the number of nodes, $m_t$ the active-edge count, $K_e$ the number of edge categories, "
        r"$d_e$ the edge-state width, and $C_\theta/M_\theta$ the time/working-memory cost of one neural "
        r"score, rate, or denoiser evaluation. Configured steps are benchmark sampling budgets, not part of the per-step asymptotics.}",
        r"\label{tab:appendix_generation_complexity}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l l c c c}",
        r"\toprule",
        r"Family & Representative & One generation step & Time / step & Working space & Configured steps \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a LaTeX table of asymptotic generation complexity. The script reads "
            "generation_complexity from train/sample metadata when available and falls back to "
            "the registered model-family specification for legacy runs."
        )
    )
    parser.add_argument("--dataset", choices=available_datasets(), default=None, help="Dataset whose metadata/configured step budget should be consulted.")
    parser.add_argument("--models", nargs="+", choices=available_models(), default=DEFAULT_MODELS)
    parser.add_argument("--run-ids", type=int, nargs="+", default=None)
    parser.add_argument("--strict-metadata", action="store_true", help="Fail if generation_complexity is absent from run metadata instead of using config fallback.")
    parser.add_argument("--output", default="outputs/tables/generation_complexity.tex")
    args = parser.parse_args()

    dataset = args.dataset or "planar"
    rows: list[dict[str, Any]] = []
    for model in args.models:
        row = _load_complexity_from_metadata(dataset, model, args.run_ids)
        if row is None:
            if args.strict_metadata:
                raise FileNotFoundError(
                    f"No generation_complexity metadata found for dataset={dataset}, model={model}. "
                    "Re-run training/sampling with the patched code or omit --strict-metadata."
                )
            row = _load_config_complexity(model, dataset=dataset)
            logger.info("Using registered/config fallback for %s/%s", dataset, model)
        rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_latex(args.models, rows) + "\n", encoding="utf-8")
    logger.info("Saved generation-complexity table to %s", out)


if __name__ == "__main__":
    main()

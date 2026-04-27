# Empirical Comparison for Time-Dependent Graph Generation

Starter scaffold for running a unified empirical comparison across representative time-dependent graph generative models on synthetic benchmarks.

## Included
- `pyproject.toml` for packaging and CLI entry points
- Config-driven experiment layout
- Script templates for data prep, training, sampling, evaluation, aggregation, and LaTeX export
- Package scaffold under `src/empirical_comparison`
- Placeholder tests

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python3 scripts/prepare_data.py --dataset sbm
python3 scripts/train_model.py --model digress --dataset sbm
python3 scripts/generate_samples.py --model digress --dataset sbm --num-samples 128
python3 scripts/evaluate_descriptor_metrics.py --model digress --dataset sbm
python3 scripts/evaluate_learned_feature_metrics.py --model digress --dataset sbm
python3 scripts/evaluate_classifier_metrics.py --model digress --dataset sbm
python3 scripts/aggregate_results.py
python3 scripts/make_latex_tables.py
```

## Notes
This scaffold is intentionally lightweight. Model wrappers currently generate placeholder outputs and should be connected to actual repositories or local implementations.

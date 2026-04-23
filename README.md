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
prepare-data --dataset sbm
train-model --model digress --dataset sbm
generate-samples --model digress --dataset sbm --num-samples 128
eval-descriptor --model digress --dataset sbm
eval-learned --model digress --dataset sbm
eval-classifier --model digress --dataset sbm
aggregate-results
make-latex-tables
```

## Notes
This scaffold is intentionally lightweight. Model wrappers currently generate placeholder outputs and should be connected to actual repositories or local implementations.

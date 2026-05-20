# Benchmark revision notes

This revision addresses the requested evaluation/protocol changes.

## Run protocol and sample/reference counts

- `scripts/run_benchmark.py` now honors `num_runs`, metric `enabled` flags, `num_generated_graphs`, `num_reference_graphs`, descriptor/feature bootstrap rounds, classifier settings, and learned-feature encoder settings from `configs/experiment.yaml`.
- Synthetic datasets use `num_runs: 3` by default; `qm9` and `zinc` are listed under `single_run_datasets` and run once by default.
- Run-aware synthetic paths are used for checkpoints, samples, training metadata, and metrics.
- `num_reference_graphs` is now enforced via `--max-reference-graphs`; generated samples are capped independently with `--max-generated-graphs`.
- `configs/datasets/sbm.yaml` and `configs/datasets/planar.yaml` now use `num_graphs: 10240`, giving about 1024 graphs in the 10% test split.

## Feature-space MMD

- `scripts/evaluate_learned_feature_metrics.py` now defaults to a fitted WL subtree encoder (`wl_subtree`) instead of the previous deterministic structural fallback.
- The previous `structural` and `random_gin` encoders remain as ablations/backward-compatible options.
- The output still includes `learned_feature_mmd` as a compatibility key, but the encoder metadata makes clear whether the representation is WL, structural, or random-feature based.

## Molecular metrics

- Added `scripts/evaluate_molecular_descriptor_metrics.py`.
- Added `src/empirical_comparison/metrics/molecular/rdkit_validity.py`.
- Molecular evaluation includes generic descriptor MMDs, atom-type MMD, bond-type MMD, RDKit sanitization validity, uniqueness, novelty, and valid-unique-novel rate.
- Validity falls back to a conservative valence check only when RDKit conversion/sanitization is unavailable or fails; the backend is recorded in the JSON payload.

## Non-finite guards

- Added `src/empirical_comparison/utils/numerics.py`.
- Training now performs finite-parameter checks after wrapper training.
- Sampling now validates generated NetworkX graph payloads.
- Added non-finite loss/gradient/parameter guards to GruM, DisCo, EDP-GNN, GraphGUIDE, ConStruct, DiGress sampling conversion, and Dummy sampling.

## Classifier / PGS-JS

- `scripts/evaluate_classifier_metrics.py` now supports run-specific samples, independent reference/generated caps, and records the balanced per-class graph count used by PGS.
- The implementation records `pgs_js_distance`, the underlying JS-divergence lower bound, true-class probability summaries, 0.5-threshold accuracy, split means/stds, selected descriptors, and classifier backends.
- The README explains the PGS-JS protocol and score formula.

## DiGress wrapper revision

- Added the uploaded upstream DiGress code under `external/DiGress`.
- Patched DiGress wrapper for CUDA device safety, finite-loss/gradient/model guards, sampling finite checks, compact logging, and runtime complexity estimation.
- Updated `configs/models/digress.yaml` to default to `gpus: 1`, `num_workers: 4`, finite guards, and quieter logs.
- Added `scripts/estimate_digress_runtime.py` for DiGress complexity and A6000-style runtime estimates.

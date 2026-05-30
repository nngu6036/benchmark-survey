# Survey Benchmark: attributed graphs, QM9/ZINC, and evaluation

This repository contains the empirical benchmark scaffold for the graph-generation survey. It supports run-aware checkpoints/samples for repeated synthetic experiments, single-run real/molecular experiments, descriptor/PGS/feature-space evaluation, molecular RDKit validity metrics, and canonical attributed-graph handling for both synthetic graphs and molecular graphs.

## 1. Installation

Core synthetic-graph experiments need only the core scientific Python stack. Molecular datasets and most model wrappers require PyTorch and PyTorch Geometric.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

External model wrappers still require their upstream repositories when applicable. Set the matching environment variable, for example `DIGRESS_REPO`, `CONSTRUCT_REPO`, `DISCO_REPO`, `GRAPHGUIDE_REPO`, `EDP_GNN_REPO`, or `GRUM_REPO`, or fill `repo_root` in the corresponding `configs/models/*.yaml` file.

## 2. Prepare datasets

Synthetic featureless datasets:

```bash
PYTHONPATH=src python scripts/prepare_data.py --dataset sbm --force
PYTHONPATH=src python scripts/prepare_data.py --dataset planar --force
```

Molecular attributed datasets:

```bash
# QM9: default config converts a 12k shuffled subset. Use --max-graphs or edit
# configs/datasets/qm9.yaml to set max_graphs: null for the full dataset.
PYTHONPATH=src python scripts/prepare_data.py \
  --dataset qm9 \
  --download-root outputs/raw_datasets/qm9 \
  --force

# QM9 uses PyG's official preprocessed archive by default. This bypasses
# RDKit raw-SDF parsing, which can fail in some RDKit/PyG environments when
# a raw molecule is parsed as None. Set prefer_preprocessed: false only if you
# specifically need PyG to rebuild QM9 from the raw SDF file.

# ZINC: default config uses PyG's 12k subset and official train/val/test splits.
# Set subset: false in configs/datasets/zinc.yaml for the full ZINC dataset.
PYTHONPATH=src python scripts/prepare_data.py \
  --dataset zinc \
  --download-root outputs/raw_datasets/zinc \
  --force
```

To inspect ZINC atom-type/category values without overwriting existing prepared
splits, use:

```bash
PYTHONPATH=src python scripts/prepare_zinc_from_smiles.py \
  --csv data/zinc250k.csv \
  --smiles-col smiles \
  --train-count 10000 \
  --val-count 1000 \
  --test-count 1000 \
  --seed 42 \
  --force
```

This prints raw/canonical `atom_type` or `node_label` counts when available and
then exits. It does not rewrite `outputs/datasets/zinc/*.pkl`, so it does not
require retraining models.

The real-dataset builders convert PyG `Data` objects into NetworkX graphs with this benchmark schema:

```yaml
node["node_label"]      # categorical atom/node label
node["feats"]           # numeric node feature vector
edge["edge_type"]       # categorical bond/edge label; 0 is reserved for dense no-edge states
edge["edge_attr"]       # numeric edge feature vector, when available
graph.graph["molecular_target"]  # optional regression target vector, not used as graph_label
```

Persisted outputs are written under `outputs/datasets/<dataset>/` by default:

```text
train.pkl
val.pkl
test.pkl
metadata.json
resolved_dataset_config.yaml
```

## 3. Train models on featureless or attributed graphs

Single model and dataset:

```bash
PYTHONPATH=src python scripts/train_model.py --dataset sbm --model dummy
PYTHONPATH=src python scripts/generate_samples.py --dataset sbm --model dummy --num-samples 1024 --force
```

Run-aware synthetic repetition example:

```bash
PYTHONPATH=src python scripts/train_model.py \
  --dataset planar \
  --model digress \
  --seed 42 \
  --run-id 0 \
  --use-run-paths

PYTHONPATH=src python scripts/generate_samples.py \
  --dataset planar \
  --model digress \
  --num-samples 1024 \
  --seed 42 \
  --run-id 0 \
  --use-run-paths \
  --force
```

```bash
for run_id in {0..2}; do
  seed=$((42 + run_id))

    PYTHONPATH=src python scripts/train_model.py \
    --dataset sbm \
    --model construct \
    --seed "$seed" \
    --run-id "$run_id" \
    --use-run-paths

  PYTHONPATH=src python scripts/generate_samples.py \
    --dataset sbm \
    --model construct \
    --num-samples 1024 \
    --seed "$seed" \
    --run-id "$run_id" \
    --use-run-paths \
    --force

  PYTHONPATH=src python scripts/train_model.py \
    --dataset planar \
    --model construct \
    --seed "$seed" \
    --run-id "$run_id" \
    --use-run-paths

  PYTHONPATH=src python scripts/generate_samples.py \
    --dataset planar \
    --model construct \
    --num-samples 1024 \
    --seed "$seed" \
    --run-id "$run_id" \
    --use-run-paths \
    --force
done
```


Run-aware outputs are stored as `outputs/checkpoints/<dataset>/<model>/run_000/...`, `outputs/samples/<dataset>/<model>/run_000.pkl`, and `outputs/metrics/<dataset>/<model>/run_000/*.json`.

Attributed molecular example:

```bash
PYTHONPATH=src python scripts/train_model.py \
  --dataset qm9 \
  --model dummy \
  --seed 42 \
  --run-id 0 \
  --use-run-paths

PYTHONPATH=src python scripts/generate_samples.py \
  --dataset qm9 \
  --model dummy \
  --num-samples 1024 \
  --seed 42 \
  --run-id 0 \
  --use-run-paths \
  --force
```

```bash
for run_id in {0..2}; do
  seed=$((42 + run_id))

  PYTHONPATH=src python scripts/train_model.py \
    --dataset zinc \
    --model edp_gnn \
    --seed "$seed" \
    --run-id "$run_id" \
    --use-run-paths

  PYTHONPATH=src python scripts/generate_samples.py \
    --dataset zinc \
    --model edp_gnn \
    --num-samples 1024 \
    --seed "$seed" \
    --run-id "$run_id" \
    --use-run-paths \
    --force

  PYTHONPATH=src python scripts/train_model.py \
    --dataset qm9 \
    --model edp_gnn \
    --seed "$seed" \
    --run-id "$run_id" \
    --use-run-paths

  PYTHONPATH=src python scripts/generate_samples.py \
    --dataset qm9 \
    --model edp_gnn \
    --num-samples 1024 \
    --seed "$seed" \
    --run-id "$run_id" \
    --use-run-paths \
    --force
done
```

`generate_samples.py` now displays a graph-level progress bar by default. Add `--no-progress` to disable it in non-interactive runs. `--draw-trajectory` saves a reference-to-sample visualization under `outputs/figures/trajectories/<dataset>/<model>_trajectory.png`.

To visually inspect generated samples:

```bash
PYTHONPATH=src python scripts/draw_generated_graphs.py \
  --dataset qm9 \
  --model disco \
  --num-graphs 16 \
  --show-node-labels
```

A complete benchmark run is configured in `configs/experiment.yaml` and executed with:

```bash
PYTHONPATH=src python scripts/run_benchmark.py
```

The default protocol uses `num_runs: 3` for synthetic datasets and `real_dataset_num_runs: 1` for `qm9` and `zinc`. Synthetic run seeds are `seed + 1000 * run_id`; real/molecular datasets keep the single-run legacy layout. The same file also makes `num_reference_graphs: 1024` real by passing `--max-reference-graphs 1024` to all metric scripts.

For expensive upstream wrappers, start with `dummy`, `construct`, or `disco` on small `--max-graphs` molecular subsets before running full experiments.

## 4. Native attributed-graph support by wrapper

- `dummy`: fits graph size/density plus empirical node-label, edge-label, node-feature, edge-feature, and graph-label marginals.
- `ConStructWrapper`: trains and samples categorical node labels and categorical edge types in the dense discrete state.
- `DisCoWrapper`: trains and samples categorical node labels and categorical edge types in the dense discrete state.
- `GraphGUIDEWrapper`: conditions on numeric node features.
- `EDPGNNWrapper`: accepts numeric node features; generated output uses benchmark fallback attributes unless the upstream model is extended.
- `DiGressWrapper`: remains primarily structural in this integration; generated attributes may be attached by empirical fallback postprocessing and are marked in sample metadata.
- `GruMWrapper`: targets GruM's generic 2D adjacency generator. For QM9, sampled adjacencies are postprocessed with a simple valence-constrained atom/bond labeller (`qm9_constrained_postprocess: true`) so molecular validity is meaningful; for other attributed molecular settings, generated attributes may still use fallback postprocessing.

For attributed datasets such as QM9 and ZINC, the benchmark still runs all wrappers, but the metadata should be checked to distinguish native attribute generation from fallback empirical attribute attachment.

## 5. Evaluate all metric families

All evaluation scripts accept the same dataset/model selection pattern. The commands below work for synthetic graphs (`sbm`, `planar`) and attributed molecular graphs (`qm9`, `zinc`) after samples have been generated.

Generic descriptor MMD metrics for non-molecular datasets:

```bash
PYTHONPATH=src python scripts/evaluate_descriptor_metrics.py \
  --dataset sbm \
  --model dummy \
  --reference-split test \
  --max-reference-graphs 1024 \
  --max-generated-graphs 1024 \
  --skip-orbit
```

Molecular descriptor metrics for QM9/ZINC:

```bash
PYTHONPATH=src python scripts/evaluate_molecular_descriptor_metrics.py \
  --dataset qm9 \
  --model disco \
  --run-id 0 \
  --reference-split test \
  --max-reference-graphs 1024 \
  --max-generated-graphs 1024 \
  --skip-orbit
```

Pass `--run-id N` to evaluate one run-specific sample such as `outputs/samples/<dataset>/<model>/run_000.pkl`; metric files are written under `outputs/metrics/<dataset>/<model>/run_000/`. Pass `--run-ids 0 1 2` to evaluate several run-specific sample files and write an aggregate JSON such as `outputs/metrics/<dataset>/<model>/descriptor_metrics.aggregate.json`; bare metric keys contain the across-run mean and `<metric>_std` keys contain the across-run population standard deviation.

The generic descriptor script reports degree MMD, clustering MMD, spectral MMD, structural-summary MMD, optional orbit MMD, and `attribute_mmd` when attributes are present. The molecular descriptor script reports those generic descriptors plus `atom_type_mmd`, `bond_type_mmd`, RDKit sanitization validity, uniqueness, novelty, and `valid_unique_novel_rate`. Uniqueness and novelty are computed on canonical RDKit SMILES of valid generated molecules; novelty compares against the training split.

Official PolyGraphScore / PGS-JS metric:

```bash
PYTHONPATH=src python scripts/evaluate_polygraphscore_official.py \
  --dataset qm9 \
  --model disco \
  --run-ids 0 1 2 \
  --max-reference-graphs 1024 \
  --max-generated-graphs 1024 \
  --num-splits 5 \
  --cv-folds 4 \
  --classifier auto \
  --skip-orbits
```

`evaluate_polygraphscore_official.py` calls the official `polygraph-benchmark` implementation and writes `polygraphscore_official.json`. It accepts the same common inputs as `evaluate_classifier_metrics.py`, including `--dataset`, `--model`, `--run-id`, `--run-ids`, `--max-reference-graphs`, `--max-generated-graphs`, `--num-splits`, `--classifier`, `--skip-orbit(s)`, descriptor bin flags, and attribute-schema flags. Some compatibility flags are accepted but ignored by the official package when that package controls the corresponding behavior internally. `--classifier auto` uses TabPFN when installed and otherwise falls back to logistic regression.

`evaluate_classifier_metrics.py` remains available as the benchmark-local PGS-style fallback/ablation and writes `classifier_metrics.json`. When both official and fallback PGS files exist, `aggregate_results.py` uses the official value for overlapping table columns such as `pgs_js_distance`.

Feature-space MMD:

```bash
PYTHONPATH=src python scripts/evaluate_learned_feature_metrics.py \
  --dataset qm9 \
  --model disco \
  --reference-split train \
  --encoder wl_subtree \
  --max-reference-graphs 1024 \
  --max-generated-graphs 1024
```

The default encoder is now a fitted Weisfeiler-Lehman subtree feature encoder (`wl_subtree`) trained only on the reference split, with optional SVD projection and attribute-aware feature components. `structural` and `random_gin` remain available as ablations/backward-compatible fallbacks, but neither should be described as a trained neural learned-feature metric.

Loop over all datasets and all models:

```bash
datasets=(sbm planar qm9 zinc)
models=(construct digress disco edp_gnn graphguide grum)

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    if [[ "$dataset" == "qm9" || "$dataset" == "zinc" ]]; then
      PYTHONPATH=src python scripts/evaluate_molecular_descriptor_metrics.py \
        --dataset "$dataset" \
        --model "$model" \
        --run-ids 0 1 2 \
        --max-reference-graphs 1024 \
        --max-generated-graphs 1024 \
        --skip-orbit
    else
      PYTHONPATH=src python scripts/evaluate_descriptor_metrics.py \
        --dataset "$dataset" \
        --model "$model" \
        --run-ids 0 1 2\
        --max-reference-graphs 1024 \
        --max-generated-graphs 1024 \
        --skip-orbit
    fi

    PYTHONPATH=src python scripts/evaluate_polygraphscore_official.py \
      --dataset "$dataset" \
      --model "$model" \
      --run-ids 0 1 2\
      --max-reference-graphs 1024 \
      --max-generated-graphs 1024 \
      --num-splits 5 \
      --cv-folds 4 \
      --classifier auto \
      --skip-orbits

    PYTHONPATH=src python scripts/evaluate_learned_feature_metrics.py \
      --dataset "$dataset" \
      --model "$model" \
      --run-ids 0 1 2\
      --reference-split train \
      --encoder wl_subtree \
      --max-reference-graphs 1024 \
      --max-generated-graphs 1024
  done
done
```

Metric files are written under `outputs/metrics/<dataset>/<model>/`.

Training and sampling metadata include hardware, runtime, normalized sampling time per 128 graphs, and peak memory fields for compute-budget reporting. After training and sampling the target rows, generate the LaTeX table with:

```bash
PYTHONPATH=src python scripts/make_compute_budget_table.py \
  --datasets planar sbm \
  --models graphguide digress construct edp_gnn disco grum \
  --run-ids 0 1 2
```

For a single compute-budget row, use `--dataset` and `--model`. `make_compute_budget_table.py` reads training metadata from `outputs/runs/<dataset>/<model>/.../train_metadata.json` and sampling metadata from `outputs/samples/<dataset>/<model>/...metadata.json`; it does not read `outputs/metrics`. Add `--debug-sources` to log the exact metadata JSON files used.

To inspect which run ids exist and whether training/sampling completed, run:

```bash
PYTHONPATH=src python scripts/report_run_status.py
PYTHONPATH=src python scripts/report_run_status.py --dataset qm9 --model grum
```

With no dataset or model arguments, `report_run_status.py` checks all benchmark datasets and models. Use `--datasets` or `--models` for multi-value subsets, and `--run-ids 0 1 2` to check specific repeated runs.

After metric evaluation, aggregate the JSON metric outputs and generate the benchmark LaTeX tables with:

```bash
PYTHONPATH=src python scripts/aggregate_results.py
PYTHONPATH=src python scripts/make_latex_tables.py
```

By default, `aggregate_results.py` uses existing `*.aggregate.json` metric files when present; otherwise it averages all discovered per-run metric JSONs for each dataset/model/metric family. To recompute tables from an explicit subset of datasets, models, or runs, pass filters:

```bash
PYTHONPATH=src python scripts/aggregate_results.py \
  --datasets planar sbm \
  --models digress construct disco grum \
  --run-ids 0 1 2
PYTHONPATH=src python scripts/make_latex_tables.py \
  --datasets planar sbm \
  --models digress construct disco grum
```

For a single LaTeX table row subset, use `--dataset` and `--model` with `make_latex_tables.py`.

If both `polygraphscore_official.json` and `classifier_metrics.json` are present for a dataset/model, `aggregate_results.py` uses the official PolyGraphScore value for overlapping PGS columns such as `pgs_js_distance`.

`make_latex_tables.py` writes a full table to `outputs/tables/aggregated_results.tex` and a simplified value-only table to `outputs/tables/aggregated_results_simple.tex`.

To generate only the molecular reporting table, run:

```bash
PYTHONPATH=src python scripts/aggregate_results.py --run-ids 0 1 2
PYTHONPATH=src python scripts/make_molecular_benchmark_table.py \
  --datasets qm9 zinc \
  --models digress construct disco grum
```

The full molecular table is written to `outputs/tables/molecular_benchmark_results.tex`, and the simplified value-only version is written to `outputs/tables/molecular_benchmark_results_simple.tex`. Missing metric values are rendered as `--`, and GraphGUIDE/EDP-GNN are intentionally omitted from the default molecular table because the current benchmark implementations do not support attributed molecular graphs.

Model hyperparameters used by the benchmark wrappers are summarized below. Values come from `configs/models/*.yaml`; public upstream defaults are used only where the wrapper keeps the upstream model shape.


## 6. Paper-style PGS-JS protocol

`scripts/evaluate_polygraphscore_official.py` is the preferred PGS path. It delegates descriptor selection, fit/test splitting, cross-validation, and scoring to the official `polygraph-benchmark` implementation, then saves `polygraphscore_official.json` under `outputs/metrics/<dataset>/<model>/`.

`scripts/evaluate_classifier_metrics.py` implements a benchmark-local balanced, held-out PGS-JS classifier protocol that is useful as a fallback or ablation:

1. Load the reference split and generated samples; apply `--max-reference-graphs` and `--max-generated-graphs`.
2. Balance the two classes by using `min(num_reference_graphs, num_generated_graphs)` graphs per class. This count is recorded as `balanced_graphs_per_class_used`.
3. For each repeated partition, randomly split reference and generated graphs into fit and held-out test halves.
4. For each descriptor, fit the descriptor on the fit half, run stratified cross-validation on the fit half, and score the held-out validation folds with the PGS-JS lower-bound objective.
5. Select the descriptor with the highest cross-validation score, fit the final classifier on the full fit half, and evaluate on the held-out test half.
6. Average `pgs_js_distance` across repeated partitions and report the split standard deviation.

For held-out examples with true domain label probability `p_true`, the local fallback computes `JSD_lb = clip(mean(log2(p_true)) + 1, 0, 1)` and reports `pgs_js_distance = sqrt(JSD_lb)`. Lower is better because a classifier that cannot distinguish generated from reference graphs gives a score near 0, while an easily separable generated distribution gives a higher score. The payload also records `pgs_js_divergence_lower_bound`, `pgs_mean_true_class_probability`, `pgs_binary_accuracy_at_0_5`, the selected descriptor, descriptor-wise CV/test scores, and the resolved classifier backend.

## 7. Useful output locations

```text
outputs/checkpoints/<dataset>/<model>/...   # trained checkpoint for each model/dataset
outputs/samples/<dataset>/...               # legacy single-run generated graph pickles and sample metadata
outputs/samples/<dataset>/<model>/run_000.pkl # run-aware synthetic samples
outputs/runs/<dataset>/<model>/...          # resolved train configs and training metadata
outputs/metrics/<dataset>/<model>/...       # legacy single-run metric payloads
outputs/metrics/<dataset>/<model>/run_000/  # run-aware synthetic metric payloads
```

## DiGress wrapper notes

The revised DiGress wrapper can use the included upstream code at `external/DiGress`. You can still override this with `DIGRESS_REPO=/path/to/DiGress` or `repo_root` in `configs/models/digress.yaml`.

Typical commands on a two-GPU server are:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_model.py --model digress --dataset sbm --run-id 0 --use-run-paths
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/generate_samples.py --model digress --dataset sbm --run-id 0 --use-run-paths --num-samples 1024 --force

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/train_model.py --model digress --dataset planar --run-id 0 --use-run-paths
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/generate_samples.py --model digress --dataset planar --run-id 0 --use-run-paths --num-samples 1024 --force
```

The current default DiGress config uses `num_epochs: 100`, `batch_size: 32`, `sample_batch_size: 32`, and `diffusion_steps: 500`. Synthetic SBM and planar configs use `num_graphs: 10240` and `num_nodes: 64`, which gives approximately 8192/1024/1024 train/validation/test graphs under the 80/10/10 split.

To print rough operation-count and runtime estimates:

```bash
PYTHONPATH=src python scripts/estimate_digress_runtime.py --dataset sbm
PYTHONPATH=src python scripts/estimate_digress_runtime.py --dataset planar
```

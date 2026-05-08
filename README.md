# Survey Benchmark: repeated runs, attributed graphs, QM9/ZINC, and evaluation

This repository contains the empirical benchmark scaffold for the graph-generation survey. It supports repeated training runs, sampling manifests, descriptor/PGS/feature-space evaluation, and canonical attributed-graph handling for both synthetic graphs and molecular graphs.

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

# ZINC: default config uses PyG's 12k subset and official train/val/test splits.
# Set subset: false in configs/datasets/zinc.yaml for the full ZINC dataset.
PYTHONPATH=src python scripts/prepare_data.py \
  --dataset zinc \
  --download-root outputs/raw_datasets/zinc \
  --force
```

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

Repeated independent runs:

```bash
PYTHONPATH=src python scripts/train_model.py \
  --dataset qm9 \
  --model disco \
  --num-runs 5

PYTHONPATH=src python scripts/generate_samples.py \
  --dataset qm9 \
  --model disco \
  --num-runs 5 \
  --num-samples 1024 \
  --force
```

`generate_samples.py` now displays a graph-level progress bar by default. Add `--no-progress` to disable it in non-interactive runs.

A complete loop over all registered datasets and models can be run as follows:

```bash
datasets=(sbm planar qm9 zinc)
models=(dummy construct digress disco edp_gnn graphguide grum)

for dataset in "${datasets[@]}"; do
  PYTHONPATH=src python scripts/prepare_data.py --dataset "$dataset" --force
  for model in "${models[@]}"; do
    PYTHONPATH=src python scripts/train_model.py \
      --dataset "$dataset" \
      --model "$model" \
      --num-runs 3
    PYTHONPATH=src python scripts/generate_samples.py \
      --dataset "$dataset" \
      --model "$model" \
      --num-runs 3 \
      --num-samples 1024 \
      --force
  done
done
```

For expensive upstream wrappers, start with `dummy`, `construct`, or `disco` on small `--max-graphs` molecular subsets before running full experiments.

## 4. Native attributed-graph support by wrapper

- `dummy`: fits graph size/density plus empirical node-label, edge-label, node-feature, edge-feature, and graph-label marginals.
- `ConStructWrapper`: trains and samples categorical node labels and categorical edge types in the dense discrete state.
- `DisCoWrapper`: trains and samples categorical node labels and categorical edge types in the dense discrete state.
- `GraphGUIDEWrapper`: conditions on numeric node features.
- `EDPGNNWrapper`: accepts numeric node features; generated output uses benchmark fallback attributes unless the upstream model is extended.
- `DiGressWrapper` and `GruMWrapper`: remain primarily structural in this integration; generated attributes may be attached by empirical fallback postprocessing and are marked in sample metadata.

For attributed datasets such as QM9 and ZINC, the benchmark still runs all wrappers, but the metadata should be checked to distinguish native attribute generation from fallback empirical attribute attachment.

## 5. Evaluate all metric families

All evaluation scripts accept the same dataset/model/run selection pattern. The commands below work for synthetic graphs (`sbm`, `planar`) and attributed molecular graphs (`qm9`, `zinc`) after samples have been generated.

Descriptor MMD metrics:

```bash
PYTHONPATH=src python scripts/evaluate_descriptor_metrics.py \
  --dataset qm9 \
  --model disco \
  --num-runs 5 \
  --reference-split test \
  --skip-orbit
```

This reports structural MMDs such as degree, clustering, spectral, structural summary, optional orbit MMD, and `attribute_mmd` when attributes are present. Use `--no-attribute-mmd` to disable the attribute descriptor.

Classifier/PolyGraphScore-style metrics:

```bash
PYTHONPATH=src python scripts/evaluate_classifier_metrics.py \
  --dataset qm9 \
  --model disco \
  --num-runs 5 \
  --num-splits 5 \
  --cv-folds 4 \
  --classifier auto \
  --skip-orbits
```

`--classifier auto` uses TabPFN when installed and otherwise falls back to standardized logistic regression. The default descriptor pool includes structural descriptors and an `attributes` descriptor when attributes are available.

Learned/fallback feature-space MMD:

```bash
PYTHONPATH=src python scripts/evaluate_learned_feature_metrics.py \
  --dataset qm9 \
  --model disco \
  --num-runs 5 \
  --reference-split train
```

The default encoder is the deterministic structural fallback encoder with attribute-aware feature components enabled. Add `--no-attribute-features` to evaluate only structural features.

Loop over all datasets and all models:

```bash
datasets=(sbm planar qm9 zinc)
models=(dummy construct digress disco edp_gnn graphguide grum)

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    PYTHONPATH=src python scripts/evaluate_descriptor_metrics.py \
      --dataset "$dataset" \
      --model "$model" \
      --num-runs 3 \
      --skip-orbit

    PYTHONPATH=src python scripts/evaluate_classifier_metrics.py \
      --dataset "$dataset" \
      --model "$model" \
      --num-runs 3 \
      --num-splits 5 \
      --cv-folds 4 \
      --classifier auto \
      --skip-orbits

    PYTHONPATH=src python scripts/evaluate_learned_feature_metrics.py \
      --dataset "$dataset" \
      --model "$model" \
      --num-runs 3 \
      --reference-split train
  done
done
```

Metric files are written under `outputs/metrics/<dataset>/<model>/`. For repeated runs, each run has its own result file and the scripts also write an aggregate JSON with mean/std across runs.

## 6. Paper-style PolyGraphScore protocol

`scripts/evaluate_classifier_metrics.py` implements a PolyGraphScore-style classifier protocol:

1. Load reference and generated graphs.
2. Randomly split each class into fit and held-out test halves.
3. For each descriptor, run stratified cross-validation on the fit half.
4. Select the descriptor with the best validation score.
5. Train the final classifier on the full fit half for the selected descriptor.
6. Report the selected descriptor's held-out test score.

The default mode is Jensen-Shannon distance, reported as `polygraphscore`, `pgs`, and `pgs_js_distance`; lower is better.

## 7. Useful output locations

```text
outputs/checkpoints/<dataset>/<model>/...   # trained checkpoints, run-aware when --num-runs > 1
outputs/samples/<dataset>/...               # generated graph pickles and sample metadata
outputs/runs/<dataset>/<model>/...          # resolved train configs and training metadata
outputs/metrics/<dataset>/<model>/...       # metric payloads and aggregate mean/std files
```

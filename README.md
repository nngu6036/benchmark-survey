# Survey Benchmark: attributed graphs, QM9/ZINC, and evaluation

This repository contains the empirical benchmark scaffold for the graph-generation survey. It supports one checkpoint per model/dataset, sample metadata, descriptor/PGS/feature-space evaluation, and canonical attributed-graph handling for both synthetic graphs and molecular graphs.

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

Attributed molecular example:

```bash
PYTHONPATH=src python scripts/train_model.py \
  --dataset qm9 \
  --model disco

PYTHONPATH=src python scripts/generate_samples.py \
  --dataset qm9 \
  --model disco \
  --num-samples 1024 \
  --draw-trajectory \
  --trajectory-graphs 2 \
  --trajectory-steps 8 \
  --force
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

A complete loop over all registered datasets and models can be run as follows:

```bash
datasets=(sbm planar qm9 zinc)
models=(dummy construct digress disco edp_gnn graphguide grum)

for dataset in "${datasets[@]}"; do
  PYTHONPATH=src python scripts/prepare_data.py --dataset "$dataset" --force
  for model in "${models[@]}"; do
    PYTHONPATH=src python scripts/train_model.py \
      --dataset "$dataset" \
      --model "$model"
    PYTHONPATH=src python scripts/generate_samples.py \
      --dataset "$dataset" \
      --model "$model" \
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

All evaluation scripts accept the same dataset/model selection pattern. The commands below work for synthetic graphs (`sbm`, `planar`) and attributed molecular graphs (`qm9`, `zinc`) after samples have been generated.

Descriptor MMD metrics:

```bash
PYTHONPATH=src python scripts/evaluate_descriptor_metrics.py \
  --dataset qm9 \
  --model disco \
  --reference-split test \
  --skip-orbit
```

For synthetic datasets, this reports structural MMDs such as degree, clustering, spectral, structural summary, optional orbit MMD, and `attribute_mmd` when attributes are present. For QM9, this script reports only validity, uniqueness, novelty, atom-type MMD, and bond-type MMD.

Classifier/PGS-JS metric:

```bash
PYTHONPATH=src python scripts/evaluate_classifier_metrics.py \
  --dataset qm9 \
  --model disco \
  --num-splits 5 \
  --cv-folds 4 \
  --classifier auto \
  --skip-orbits
```

`--classifier auto` uses TabPFN when installed and otherwise falls back to standardized logistic regression. The script reports only `pgs_js_distance`; the default descriptor pool includes structural descriptors and an `attributes` descriptor when attributes are available.

Learned/fallback feature-space MMD:

```bash
PYTHONPATH=src python scripts/evaluate_learned_feature_metrics.py \
  --dataset qm9 \
  --model disco \
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
      --skip-orbit

    PYTHONPATH=src python scripts/evaluate_classifier_metrics.py \
      --dataset "$dataset" \
      --model "$model" \
      --num-splits 5 \
      --cv-folds 4 \
      --classifier auto \
      --skip-orbits

    PYTHONPATH=src python scripts/evaluate_learned_feature_metrics.py \
      --dataset "$dataset" \
      --model "$model" \
      --reference-split train
  done
done
```

Metric files are written under `outputs/metrics/<dataset>/<model>/`.

Training and sampling metadata include hardware, runtime, normalized sampling time per 128 graphs, and peak memory fields for compute-budget reporting. After training and sampling the target rows, generate the LaTeX table with:

```bash
PYTHONPATH=src python scripts/make_compute_budget_table.py \
  --datasets planar sbm \
  --models graphguide digress construct edp_gnn disco grum
```

Model hyperparameters used by the benchmark wrappers are summarized below. Values come from `configs/models/*.yaml`; public upstream defaults are used only where the wrapper keeps the upstream model shape.

```latex
\begin{table}[http]
\centering
\caption{Model-specific hyperparameters used in the benchmark. Values follow the benchmark model configs, with public implementation defaults used for model-shape fields left unset by the wrapper.}
\label{tab:appendix_model_hyperparameters}
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{l c c c c c c c}
\toprule
Model & Hidden dim. & Layers & Optimizer & Learning rate & Epochs & Noise / path schedule & Checkpoint rule \\
\midrule
GraphGUIDE &
$256$; GAT hidden $32$, $8$ heads &
$4$ GNN layers &
Adam &
$10^{-3}$ &
$100$ &
Bernoulli zero-skip; $t_{\max}=1000$, $a=100$, $b=10$ &
Final checkpoint after training \\
DiGress &
$d_X=256$, $d_E=64$, $d_y=64$; MLP $(256,128,128)$ &
$5$ transformer layers &
AdamW &
$2\times10^{-4}$ &
$100$ &
Discrete marginal diffusion; cosine schedule; $T=500$ &
Final Lightning checkpoint saved by wrapper \\
ConStruct &
$d_X=256$, $d_E=64$, $d_y=64$; MLP $(256,128,128)$ &
$5$ transformer layers &
AdamW &
$2\times10^{-4}$ &
$100$ &
Absorbing-edge discrete diffusion; $T=500$ &
Final checkpoint after training \\
EDP-GNN &
GIN hidden $(16,16,16,16)$; channels $(2,4,4,4,2)$ &
$4$ GIN blocks &
Adam + exponential LR decay &
$10^{-3}$ &
$100$ &
Score noise $\sigma\in\{0.1,0.2,0.4,0.6,0.8,1.6\}$; Langevin $1000$ steps &
Final checkpoint after training \\
DisCo &
$n_{\mathrm{dim}}=128$; GT $(d_X,d_E,d_y)=(256,64,64)$ &
$5$ graph-transformer layers &
AdamW &
$2\times10^{-4}$ &
$100$ &
Marginal CTMC; $t_{\min}=0.01$; $50$ sampling steps; $\beta=2.0$, $\alpha=0.8$ &
Best validation loss, or best training loss without validation \\
GruM &
$d_X=256$, $d_E=64$, $d_y=64$; MLP $(128,64,128)$ &
$8$ transformer layers &
AdamW &
$2\times10^{-4}$ &
$100$ &
OU bridge, $1000$ scales; $X:\sigma_0=0.2,\sigma_1=0.1$; $A:\sigma_0=0.4,\sigma_1=0.2$; Euler TV sampler &
Final checkpoint with EMA state \\
\bottomrule
\end{tabular}
}
\end{table}
```

## 6. Paper-style PGS-JS protocol

`scripts/evaluate_classifier_metrics.py` implements a PGS-JS classifier protocol:

1. Load reference and generated graphs.
2. Randomly split each class into fit and held-out test halves.
3. For each descriptor, run stratified cross-validation on the fit half.
4. Select the descriptor with the best validation score.
5. Train the final classifier on the full fit half for the selected descriptor.
6. Report the selected descriptor's held-out PGS-JS distance.

The only reported benchmark metric is `pgs_js_distance`; lower is better.

## 7. Useful output locations

```text
outputs/checkpoints/<dataset>/<model>/...   # trained checkpoint for each model/dataset
outputs/samples/<dataset>/...               # generated graph pickles and sample metadata
outputs/runs/<dataset>/<model>/...          # resolved train configs and training metadata
outputs/metrics/<dataset>/<model>/...       # metric payloads
```

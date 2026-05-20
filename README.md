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
  --dataset sbm \
  --model dummy \
  --seed 42 \
  --run-id 0 \
  --use-run-paths

PYTHONPATH=src python scripts/generate_samples.py \
  --dataset sbm \
  --model dummy \
  --num-samples 1024 \
  --seed 42 \
  --run-id 0 \
  --use-run-paths \
  --force
```

Run-aware outputs are stored as `outputs/checkpoints/<dataset>/<model>/run_000/...`, `outputs/samples/<dataset>/<model>/run_000.pkl`, and `outputs/metrics/<dataset>/<model>/run_000/*.json`.

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
- `DiGressWrapper` and `GruMWrapper`: remain primarily structural in this integration; generated attributes may be attached by empirical fallback postprocessing and are marked in sample metadata.

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
  --reference-split test \
  --max-reference-graphs 1024 \
  --max-generated-graphs 1024 \
  --skip-orbit
```

The generic descriptor script reports degree MMD, clustering MMD, spectral MMD, structural-summary MMD, optional orbit MMD, and `attribute_mmd` when attributes are present. The molecular descriptor script reports those generic descriptors plus `atom_type_mmd`, `bond_type_mmd`, RDKit sanitization validity, uniqueness, novelty, and `valid_unique_novel_rate`. Uniqueness and novelty are computed on canonical RDKit SMILES of valid generated molecules; novelty compares against the training split.

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

`--classifier auto` uses TabPFN when installed and otherwise falls back to standardized logistic regression. The main reported value is `pgs_js_distance`; the payload also stores the JS-divergence lower bound, selected descriptor, split-level scores, feature dimensions, and resolved classifier. The default descriptor pool includes structural descriptors and an `attributes` descriptor when attributes are available.

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
models=(dummy construct digress disco edp_gnn graphguide grum)

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    if [[ "$dataset" == "qm9" || "$dataset" == "zinc" ]]; then
      PYTHONPATH=src python scripts/evaluate_molecular_descriptor_metrics.py \
        --dataset "$dataset" \
        --model "$model" \
        --max-reference-graphs 1024 \
        --max-generated-graphs 1024 \
        --skip-orbit
    else
      PYTHONPATH=src python scripts/evaluate_descriptor_metrics.py \
        --dataset "$dataset" \
        --model "$model" \
        --max-reference-graphs 1024 \
        --max-generated-graphs 1024 \
        --skip-orbit
    fi

    PYTHONPATH=src python scripts/evaluate_classifier_metrics.py \
      --dataset "$dataset" \
      --model "$model" \
      --max-reference-graphs 1024 \
      --max-generated-graphs 1024 \
      --num-splits 5 \
      --cv-folds 4 \
      --classifier auto \
      --skip-orbits

    PYTHONPATH=src python scripts/evaluate_learned_feature_metrics.py \
      --dataset "$dataset" \
      --model "$model" \
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
  --models graphguide digress construct edp_gnn disco grum
```

After metric evaluation, aggregate the JSON metric outputs and generate the benchmark LaTeX tables with:

```bash
PYTHONPATH=src python scripts/aggregate_results.py
PYTHONPATH=src python scripts/make_latex_tables.py
```

To generate only the QM9 molecular reporting table, run:

```bash
PYTHONPATH=src python scripts/aggregate_results.py
PYTHONPATH=src python scripts/make_molecular_benchmark_table.py \
  --dataset qm9 \
  --models digress construct disco grum
```

The molecular table is written to `outputs/tables/qm9_benchmark_results.tex`. Missing metric values are rendered as `--`, and GraphGUIDE/EDP-GNN are intentionally omitted from the default QM9 table because the current benchmark implementations do not support attributed molecular graphs.

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

`scripts/evaluate_classifier_metrics.py` implements a balanced, held-out PGS-JS classifier protocol:

1. Load the reference split and generated samples; apply `--max-reference-graphs` and `--max-generated-graphs`.
2. Balance the two classes by using `min(num_reference_graphs, num_generated_graphs)` graphs per class. This count is recorded as `balanced_graphs_per_class_used`.
3. For each repeated partition, randomly split reference and generated graphs into fit and held-out test halves.
4. For each descriptor, fit the descriptor on the fit half, run stratified cross-validation on the fit half, and score the held-out validation folds with the PGS-JS lower-bound objective.
5. Select the descriptor with the highest cross-validation score, fit the final classifier on the full fit half, and evaluate on the held-out test half.
6. Average `pgs_js_distance` across repeated partitions and report the split standard deviation.

For held-out examples with true domain label probability `p_true`, the script computes `JSD_lb = clip(mean(log2(p_true)) + 1, 0, 1)` and reports `pgs_js_distance = sqrt(JSD_lb)`. Lower is better because a classifier that cannot distinguish generated from reference graphs gives a score near 0, while an easily separable generated distribution gives a higher score. The payload also records `pgs_js_divergence_lower_bound`, `pgs_mean_true_class_probability`, `pgs_binary_accuracy_at_0_5`, the selected descriptor, descriptor-wise CV/test scores, and the resolved classifier backend.

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

# Survey Benchmark Update: PolyGraphScore and Attributed Graphs

This repository contains the graph-generation survey benchmark with repeated-run training, model-specific wrappers, PolyGraphScore-style classifier evaluation, and canonical attributed-graph support.

## Paper-style PolyGraphScore

`scripts/evaluate_classifier_metrics.py` now implements the PolyGraphScore protocol:

1. Load reference graphs and generated graphs.
2. Randomly split each class into fit and held-out test halves.
3. For each descriptor, run stratified cross-validation on the fit half.
4. Select the descriptor with the largest validation score.
5. Train the final classifier on the full fit half for that selected descriptor.
6. Report the selected descriptor's held-out test score.

The default mode is Jensen-Shannon distance:

```bash
PYTHONPATH=src python scripts/evaluate_classifier_metrics.py \
  --dataset sbm \
  --model dummy \
  --num-splits 5 \
  --cv-folds 4 \
  --classifier auto \
  --skip-orbits
```

`--classifier auto` uses TabPFN when it is installed and otherwise uses a standardized logistic-regression fallback. The script reports `polygraphscore`, `pgs`, and `pgs_js_distance`; lower is better.

Generic descriptors are:

```text
degree, clustering, spectral, orbit4, orbit5, gin, attributes
```

ORCA-based orbit descriptors can be disabled with `--skip-orbits`.

## Repeated trained models

For each model, independent versions can be trained and evaluated:

```bash
PYTHONPATH=src python scripts/train_model.py --dataset sbm --model dummy --num-runs 5
PYTHONPATH=src python scripts/generate_samples.py --dataset sbm --model dummy --num-runs 5 --num-samples 1024 --force
PYTHONPATH=src python scripts/evaluate_classifier_metrics.py --dataset sbm --model dummy --num-runs 5 --num-splits 5 --skip-orbits
```

The benchmark stores per-run metrics and across-run aggregate mean/std metrics.

## Attributed graphs

The benchmark canonicalizes NetworkX attributes into this schema:

```yaml
graph_attributes:
  enabled: auto
  node_feature_attr: feats
  node_label_attr: node_label
  edge_feature_attr: edge_attr
  edge_label_attr: edge_type
  graph_label_attr: graph_label
  node_label_aliases: [label, node_type, type, atom_type]
  node_feature_aliases: [feature, features, x]
  edge_label_aliases: [label, edge_label, bond_type, type]
  edge_feature_aliases: [feature, features, weight]
  graph_label_aliases: [y, label, target]
  add_default_node_features: true
  add_default_node_labels: true
  add_default_edge_labels: true
  default_node_feature_dim: 1
```

Internally:

- `node["node_label"]` is a categorical node label index.
- `node["feats"]` is a fixed-width numeric node feature vector.
- `edge["edge_type"]` is a categorical edge label index; `0` is reserved for the dense no-edge class.
- `edge["edge_attr"]` is a fixed-width numeric edge feature vector.
- `graph.graph["graph_label"]` is an optional graph-level label index.

## Native wrapper support

- `dummy`: samples node labels, edge labels, numeric node features, and numeric edge features from fitted marginals.
- `ConStructWrapper`: trains/samples categorical node labels and categorical edge types in the dense discrete state.
- `DisCoWrapper`: trains/samples categorical node labels and categorical edge types in the dense discrete state.
- `GraphGUIDEWrapper`: supports numeric node features as conditioning variables.
- `EDPGNNWrapper`: accepts numeric node features; generated output uses fallback feature attributes unless the upstream model is extended.
- `DiGressWrapper` and `GruMWrapper`: remain primarily structural in this benchmark integration; generated attributes can be attached by the benchmark empirical fallback and are marked in metadata.

## Attribute-aware metrics

- PGS includes an `attributes` descriptor when attributes are present.
- `evaluate_descriptor_metrics.py` computes `attribute_mmd` unless disabled with `--no-attribute-mmd`.
- `evaluate_learned_feature_metrics.py` includes attribute-aware feature components by default and can disable them with `--no-attribute-features`.
- Learned-feature MMD defaults to `--reference-split train`, so it compares generated graph representations against training-data graph representations by default.

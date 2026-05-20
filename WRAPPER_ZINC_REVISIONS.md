# Wrapper and ZINC revisions

This revision reviews and patches the benchmark adapters for ConStruct, GruM, DisCo, and DiGress against the uploaded upstream repositories.  The upstream repositories are included under `external/` so the wrappers have a default repo root without requiring environment variables.

## External codebases included

- `external/ConStruct`
- `external/GruM`
- `external/DisCo`
- `external/DiGress`

The wrapper still accepts `CONSTRUCT_REPO`, `GRUM_REPO`, `DISCO_REPO`, and `DIGRESS_REPO` for overriding these paths.

## ConStruct

- Switched core imports from generic `models.*` / `diffusion.*` to fully qualified `ConStruct.models.*` / `ConStruct.diffusion.*` imports to reduce collisions with other uploaded repositories.
- Added a runtime patch for the upstream reverse projector so rejected-edge one-hot tensors are created on the same device and dtype as the sampled edge tensor.  This fixes CUDA sampling failures for planar reverse projection.
- Patched the copied upstream `external/ConStruct/ConStruct/projector/projector_utils.py` for the same device issue.
- Added post-optimizer finite-parameter checks and sample graph finite checks.
- Keeps categorical `node_label` and `edge_type` classes, so ZINC can be trained as a categorical molecular graph dataset.  Structural features remain disabled for molecular datasets when configured.

## GruM

- Added a successful-update counter.  Training now refuses to save a checkpoint if every optimizer update was skipped because of non-finite loss/gradients.
- Added sampling diagnostics for non-finite values before the existing `nan_to_num` stabilization.
- Added finite checks on sampled NetworkX graphs.
- GruM remains structure-first in this benchmark.  For ZINC/QM9, `scripts/generate_samples.py` now applies empirical molecular labels when a structure-only model did not natively generate `node_label`/`edge_type`, even if it did produce dummy node features.

## DisCo

- Fixed repeated epoch shuffling by seeding the epoch shuffle with `seed + epoch` rather than the same seed every epoch.
- Added successful-update tracking in training.
- Checkpoints are now saved only when the selected score is finite.  Training fails if no finite checkpoint is produced.
- Added finite checks on sampled NetworkX graphs.
- Added `disable_structural_features_for_molecular: true` to the model config and wrapper logic so ZINC/QM9 avoid SPECTRE structural auxiliary features by default.
- DisCo keeps categorical `node_label` and `edge_type` classes, so it can train/sample ZINC as a categorical molecular graph dataset.

## DiGress

- Kept runtime patches for upstream DiGress CUDA/device sampling issues.
- Added lazy dependency checks so the benchmark registry can import DiGress even if `omegaconf` or `pytorch_lightning` is not installed; training/sampling still requires those packages.
- Fixed sampling `batch_id` to increment per batch rather than by batch size.
- Generalized a QM9-only error message to molecular datasets.
- DiGress supports ZINC through the benchmark molecular materialization path, using categorical `node_label` and `edge_type` classes.

## ZINC/RDKit handling

PyG ZINC exposes categorical `atom_type` and bond categories.  The benchmark now treats ZINC integer node labels as categorical ids, not atomic numbers.  RDKit validity will use:

1. explicit node attributes such as `atomic_number`, `atomic_num`, or `z`, or
2. a user-supplied `configs/datasets/zinc.yaml:rdkit_atomic_number_mapping`, or
3. symbol-valued raw label vocabularies such as `C`, `N`, `O`, `Cl`.

Bare integer ZINC category ids are not interpreted as atomic numbers.  This avoids false-positive RDKit validity scores.

Example mapping format if you know the ZINC atom-type decoder:

```yaml
rdkit_atomic_number_mapping:
  0: 6
  1: 7
  2: 8
  3: 9
  4: 15
  5: 16
  6: 17
```

or:

```yaml
rdkit_atomic_number_mapping: [C, N, O, F, P, S, Cl]
```

Without this mapping, ZINC molecular evaluation still reports structural MMD, atom-type MMD, bond-type MMD, uniqueness/novelty where canonical SMILES are available, and a note that RDKit validity could not safely decode integer category ids.

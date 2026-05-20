# DiGress wrapper revision notes

This revision integrates the uploaded upstream DiGress code under `external/DiGress` and updates the benchmark wrapper at:

```text
src/empirical_comparison/models/wrappers/digress.py
```

## Main wrapper changes

- The wrapper defaults to `external/DiGress` when `DIGRESS_REPO` / `repo_root` are not set.
- Runtime patches fix device mismatches in upstream helper functions that create CPU `torch.eye(...)` tensors inside CUDA training/sampling paths.
- The SPECTRE dataset processing patch remains active: it prevents `SpectreGraphDataset.process` from appending every graph twice.
- Training now uses a Lightning finite-guard callback:
  - checks train/validation losses,
  - checks gradients before optimizer steps,
  - checks model parameters/buffers periodically,
  - fails if training finishes with zero optimizer steps.
- Sampling checks:
  - loaded model finite before sampling,
  - raw DiGress sample tensors finite after every sample batch,
  - converted NetworkX graphs finite before return.
- Upstream CUDA memory-summary printing is suppressed by default to avoid huge logs.
- `configs/models/digress.yaml` now defaults to GPU execution (`gpus: 1`) with 4 dataloader workers and fail-fast numerical guards.
- A helper script was added:

```bash
PYTHONPATH=src python scripts/estimate_digress_runtime.py --dataset sbm
PYTHONPATH=src python scripts/estimate_digress_runtime.py --dataset planar
```

## Important runtime note

For synthetic SPECTRE-style datasets, DiGress is dense in the number of nodes: cost scales roughly as `O(B L n^2 d_e d_x)` per forward pass and `O(B_s T L n^2 d_e d_x)` for sampling. Since SBM and planar use the same `num_nodes: 64`, their runtimes should be similar under the current config.

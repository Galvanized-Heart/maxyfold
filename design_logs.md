# Structure logging MVP

## Goal
Add local-first structure/metric logging for PDB validation without uploading CIFs to W&B.

## Must have
- [x] Masked coordinate metrics work on padded batches
- [x] PDB training logs scalar structure metrics
- [x] Local prediction bundles are saved for selected validation examples
- [x] Bundles contain `pred.cif`, `metadata.json`, `metrics.json`
- [x] `true.cif` is not duplicated
- [x] Metadata contains `pdb_id` and enough info to recover true structure later
- [x] W&B upload of CIFs is disabled by default
- [x] Code runs with `logger=null`
- [ ] Code runs with `logger=wandb`
- [ ] Tests cover metrics and writer behavior

## Out of scope
- rendered overlays
- full diffusion trajectories
- pLDDT / PAE / pTM / ipTM heads
- uploading all CIFs to W&B
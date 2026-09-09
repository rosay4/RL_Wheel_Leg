# Experiment data and results

This directory contains the CSV outputs and figures used to analyze the
wheel-leg experiments.

## Layout

- `raw/`: CSV outputs copied from the original `analysis` directory.
  - `raw/` contains the original analysis-root CSV files.
  - `raw/new_experiment/` contains the later experiment batch.
- `plots/`: generated figures and the CSV sidecars stored with them in the
  original analysis directory.
- `manifest.yaml`: source and inventory information for this snapshot.

The original filenames are preserved for traceability. Files with names such
as `(1)` or `_copy` were retained because their contents differ from the
similarly named files; they are separate experiment variants rather than
assumed duplicates.

The plotting and evaluation scripts live under `scripts/rsl_rl/`. The files in
this directory are recorded results, not the source of truth for rerunning an
experiment.

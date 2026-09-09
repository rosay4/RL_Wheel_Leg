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

## 中文说明

这里保存实验产生的 CSV 和结果图，不是重新运行实验所需的源代码。`raw/` 保存原始分析数据，`raw/new_experiment/` 保存后续实验批次，`plots/` 保存结果图及其 CSV sidecar。

带有 `(1)` 或 `_copy` 的文件内容与同名文件不同，因此保留为独立实验版本。文件来源和数量统计见 `manifest.yaml`。

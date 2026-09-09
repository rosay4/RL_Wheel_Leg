# Experiment data and results

This directory contains the CSV outputs and figures used to analyze the
wheel-leg experiments.

## Layout

- `raw/`: CSV evaluation and analysis outputs.
  - `raw/` contains evaluation, terrain, impulse-response, and parameter-shift data.
  - `raw/new_experiment/` contains the later experiment batch.
- `plots/`: generated figures and their CSV sidecars.
- `manifest.yaml`: source and inventory information for this snapshot.

Filename suffixes such as `(1)` and `_copy` distinguish separate result
variants.

The plotting and evaluation scripts live under `scripts/rsl_rl/`. The files in
this directory are recorded results, not the source of truth for rerunning an
experiment.

## 中文说明

这里保存实验产生的 CSV 和结果图，不是重新运行实验所需的源代码。`raw/` 保存原始分析数据，`raw/new_experiment/` 保存后续实验批次，`plots/` 保存结果图及其 CSV sidecar。

文件名中的 `(1)` 或 `_copy` 用于区分不同的结果版本。文件统计见 `manifest.yaml`。

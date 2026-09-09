# Published checkpoints

This directory contains a small, curated set of checkpoints selected from the
original training archive. It is not a copy of every training run.

Each published run contains:

- one selected RSL-RL checkpoint (`model_*.pt`);
- exported TorchScript and ONNX policies;
- the environment and agent YAML snapshots written by the training script.

Training logs, intermediate checkpoints, pickle snapshots, TensorBoard event
files, and internal Git diffs are intentionally omitted. The source archive was
`models.zip`; the exact source run for each published checkpoint is recorded in
`manifest.yaml`.

The exported policy files are the recommended artifacts for inference. The
`.pt` checkpoint files are provided for resuming or inspecting training.

## 中文说明

这里保存从原始训练结果中筛选出的代表性模型，而不是全部训练过程。每个模型目录包含一个选定的 RSL-RL checkpoint、ONNX 策略、TorchScript 策略，以及训练时保存的环境和 agent 配置快照。

模型来源、任务对应关系和待确认项见 `manifest.yaml`。其中任务映射不明确的模型会明确标记为 `null`，不会根据文件名擅自推断。

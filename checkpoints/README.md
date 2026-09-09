# Published checkpoints

This directory contains a small, curated set of checkpoints selected from the
original training archive. It is not a copy of every training run.

Each published run contains:

- one selected RSL-RL checkpoint (`model_*.pt`);
- exported TorchScript and ONNX policies;
- the environment and agent YAML snapshots written by the training script.

The exact source run for each published checkpoint is recorded in
`manifest.yaml`.

The exported policy files are the recommended artifacts for inference. The
`.pt` checkpoint files are provided for resuming or inspecting training.

## 中文说明

这里保存一组代表性模型。每个模型目录包含一个选定的 RSL-RL checkpoint、ONNX 策略、TorchScript 策略，以及对应的环境和 agent 配置快照。

模型来源和任务对应关系见 `manifest.yaml`。

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

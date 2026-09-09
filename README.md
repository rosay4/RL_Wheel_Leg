# RL Wheel Leg

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-compatible-orange.svg)](https://isaac-sim.github.io/IsaacLab/)

基于 NVIDIA Isaac Lab 的轮腿机器人强化学习仿真与控制项目，包含机器人资产、强化学习环境、RSL-RL PPO 训练与评估脚本、策略导出、代表性 checkpoint 以及实验结果。

## 项目简介

本项目面向轮腿机器人在多种地形下的运动控制研究，重点覆盖速度跟踪、姿态稳定、轮腿协同控制和复杂地形通过能力。项目同时整理了从 SolidWorks、URDF 到 USD 的机器人模型链路，便于复现实验和后续 Sim2Real 开发。

主要内容：

- Isaac Lab Manager-Based RL 环境；
- 平地、坡面、粗糙地形、混合地形和非结构化地形任务；
- 轮腿机器人 SolidWorks、URDF、STL 和 USD 资产；
- RSL-RL PPO 训练、评估、播放和策略导出脚本；
- ONNX、TorchScript 和 RSL-RL checkpoint；
- 速度跟踪、参数扰动、冲击响应和地形穿越实验数据。

## 项目结构

```text
.
├── assets/                 # URDF、STL、USD 仿真资产
├── cad/                    # SolidWorks 零件和装配体
├── checkpoints/            # 精选训练模型和模型清单
├── configs/                # 任务和配置索引
├── experiments/            # 原始实验 CSV 和结果图
├── scripts/                # 训练、评估、播放和绘图脚本
├── source/wheel_leg/       # Isaac Lab 扩展和任务代码
├── LICENSE                 # Apache-2.0 许可证
├── NOTICE                  # 第三方组件和资产说明
└── CITATION.cff            # 软件引用信息
```

## 环境要求

运行本项目需要一个与代码兼容的 Isaac Sim、Isaac Lab、PyTorch、CUDA 和 RSL-RL 环境。

请先按照 [Isaac Lab 安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。公开复现实验时，建议记录：

- Isaac Sim 和 Isaac Lab 版本；
- Python、PyTorch 和 CUDA 版本；
- GPU 型号；
- 任务 ID、随机种子和 checkpoint；
- 训练或评估使用的命令行参数。

## 安装

在已经配置好 Isaac Lab 的 Python 环境中执行：

```bash
python -m pip install -e source/wheel_leg
```

如果使用 Isaac Lab 提供的启动脚本，请将下面命令中的 `python` 替换为对应的 Isaac Lab Python 启动方式。

安装后检查任务是否成功注册：

```bash
python scripts/list_envs.py
```

## 快速开始

### 运行基础环境

```bash
python scripts/rsl_rl/train.py --task=Template-Wheel-Leg-v0
```

### 使用零动作或随机动作检查环境

```bash
python scripts/zero_agent.py --task=Template-Wheel-Leg-v0
python scripts/random_agent.py --task=Template-Wheel-Leg-v0
```

### 播放已发布策略

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Wheel-Leg-v0 \
  --checkpoint=checkpoints/flat_cmd_v1_1/model_999.pt
```

训练日志默认写入 `logs/rsl_rl/<experiment_name>/`。日志目录仅用于本地运行，不属于版本化实验资产。

## 任务列表

任务 ID 和配置类的完整对应关系见 [configs/task_catalog.yaml](configs/task_catalog.yaml)。

| 任务 ID | 用途 |
|---|---|
| `Template-Wheel-Leg-v0` | 平地训练 |
| `Template-Wheel-Leg-Rough-v0` | 粗糙地形训练 |
| `Template-Wheel-Leg-Slope-v0` | 坡面训练或评估 |
| `Template-Wheel-Leg-Mixed-v0` | 混合地形训练或评估 |
| `Template-Wheel-Leg-Unstructured-v0` | 非结构化地形训练 |
| `Template-Wheel-Leg-Unstructured-NoEnergyPenalty-v0` | 无能耗惩罚消融实验 |
| `Template-Wheel-Leg-Unstructured-HighPosturePenalty-v0` | 高姿态惩罚消融实验 |
| `Template-Wheel-Leg-NoDr-v0` | 无域随机化消融实验 |
| `Template-Wheel-Leg-Eval-Stairs-v0` | 台阶评估 |
| `Template-Wheel-Leg-Eval-Rough-v0` | 崎岖地形评估 |

## 机器人资产

- [assets/wheel_leg/](assets/wheel_leg/)：仿真使用的 URDF、STL 和 USD；
- [cad/solidworks/](cad/solidworks/)：SolidWorks 零件和装配体；
- `source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/wheel_leg_env_cfg.py`：机器人 USD 和仿真环境配置。

SolidWorks 装配体依赖同目录下的零件文件，使用时请保持目录结构不变。仿真环境会从仓库内的 `assets/wheel_leg/usd/` 加载机器人模型。

## Checkpoint

[checkpoints/](checkpoints/) 提供经过筛选的代表性训练结果。每个模型目录包含：

- 一个选定的 RSL-RL checkpoint；
- ONNX 策略；
- TorchScript 策略；
- 环境和 agent 配置快照。

模型来源、任务映射和文件路径见 [checkpoints/manifest.yaml](checkpoints/manifest.yaml)。

## 实验结果

[experiments/](experiments/) 保存实验 CSV 和结果图，包含：

- 速度跟踪；
- 参数扰动和域随机化；
- 冲击响应；
- 坡面姿态与轨迹；
- 崎岖地形和台阶通过；
- 奖励项、训练曲线和扭矩消融结果。

实验数据组织方式见 [experiments/README.md](experiments/README.md)，绘图和评估脚本位于 `scripts/rsl_rl/`。

## 代码入口

- 环境与奖励：`source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/wheel_leg_env_cfg.py`；
- PPO 配置：`source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/agents/rsl_rl_ppo_cfg.py`；
- 任务注册：`source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/__init__.py`；
- 训练：`scripts/rsl_rl/train.py`；
- 播放与导出：`scripts/rsl_rl/play.py`、`scripts/rsl_rl/play_keyboard.py`；
- 评估：`scripts/rsl_rl/eval_*.py`；
- 绘图：`scripts/rsl_rl/plot_*.py`。

## 开发

安装代码检查工具：

```bash
pip install pre-commit
pre-commit run --all-files
```

## 许可证

本项目源代码采用 [Apache License 2.0](LICENSE)。Isaac Lab 相关模板代码保留其原有版权和 BSD-3-Clause 许可声明，具体说明见 [NOTICE](NOTICE)。

CAD、USD、模型权重和其他第三方资产可能具有独立的许可证或再分发限制，使用前请遵循其对应的许可条款。

## 引用

如果本项目对你的研究有帮助，请参考 [CITATION.cff](CITATION.cff) 引用。

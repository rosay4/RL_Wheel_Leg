# 轮腿机器人强化学习仿真项目

这是一个基于 NVIDIA Isaac Lab 的轮腿机器人强化学习项目，包含机器人仿真环境、训练与评估脚本、URDF/USD/CAD 资产、代表性策略模型以及实验结果。

项目的主要目标是研究轮腿机器人在平地、坡面、崎岖地形和非结构化地形中的运动控制与速度跟踪能力，并为后续 Sim2Real 部署提供策略和接口参考。

> 项目仍在持续整理中。当前仓库主要覆盖仿真、强化学习训练、策略导出和实验数据，不保证所有实机底层控制代码都包含在内。

## 项目内容

- Isaac Lab Manager-Based RL 环境；
- 轮腿机器人 USD、URDF、STL 和 SolidWorks 资产；
- RSL-RL PPO 训练与评估脚本；
- 平地、坡面、混合地形、粗糙地形和非结构化地形任务；
- ONNX、TorchScript 和 RSL-RL checkpoint；
- 速度跟踪、参数扰动、冲击响应、坡面和地形穿越实验结果；

## 仓库结构

```text
.
├── assets/                 # URDF、STL、USD 仿真资产
├── cad/                    # SolidWorks 零件和装配体
├── checkpoints/            # 精选训练模型和模型清单
├── configs/                # 任务和配置索引
├── experiments/            # 原始实验 CSV 和结果图
├── scripts/                # 训练、评估、播放和绘图脚本
├── source/wheel_leg/       # Isaac Lab 扩展和任务代码
└── plots/                  # 可选的图表输出目录
```

## 环境要求

运行项目需要安装与项目代码兼容的 Isaac Sim、Isaac Lab、PyTorch、CUDA 和 RSL-RL 环境。请先参考 [Isaac Lab 安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)。

建议在公开复现实验时记录以下版本信息：

- Isaac Sim 版本；
- Isaac Lab 版本；
- Python 版本；
- PyTorch 和 CUDA 版本；
- GPU 型号和显存；
- 使用的任务 ID、随机种子和 checkpoint。

当前仓库没有把 Isaac Lab 整体复制进来，也没有把本机 Python 环境提交到仓库。

## 安装项目扩展

在已经配置好 Isaac Lab 的 Python 环境中执行：

```bash
python -m pip install -e source/wheel_leg
```

如果没有使用 Isaac Lab 的 Python 环境，请将下面命令中的 `python` 替换成对应的 Isaac Lab 启动脚本，例如 `./isaaclab.sh -p` 或 Windows 环境下的等效命令。

## 检查任务注册

```bash
python scripts/list_envs.py
```

任务名称和配置类的对应关系见 [configs/task_catalog.yaml](configs/task_catalog.yaml)。

当前已注册的主要任务包括：

| 任务 | 用途 |
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

## 训练

基本训练命令：

```bash
python scripts/rsl_rl/train.py --task=Template-Wheel-Leg-v0
```

常用参数示例：

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Wheel-Leg-v0 \
  --num_envs=4096 \
  --max_iterations=1000 \
  --seed=42
```

训练日志默认写入 `logs/rsl_rl/<experiment_name>/`。`logs/` 已被 Git 忽略，不应提交到公开仓库。

零动作和随机动作可以用于检查环境是否能正常初始化：

```bash
python scripts/zero_agent.py --task=Template-Wheel-Leg-v0
python scripts/random_agent.py --task=Template-Wheel-Leg-v0
```

## 播放和策略导出

播放 checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Wheel-Leg-v0 \
  --checkpoint=checkpoints/flat_cmd_v1_1/model_999.pt
```

播放脚本可以导出 TorchScript 和 ONNX 策略。已经导出的代表性策略位于 [checkpoints/](checkpoints/)。

## 机器人资产

- [assets/wheel_leg/](assets/wheel_leg/)：仿真使用的 URDF、STL 和 USD；
- [cad/solidworks/](cad/solidworks/)：SolidWorks 零件和装配体；
- 仿真配置会自动从仓库内的 `assets/wheel_leg/usd/` 查找 USD，不依赖个人电脑或云桌面路径。

SolidWorks 装配体依赖同目录下的零件文件，打开装配体时请保持目录结构不变。部分零部件可能来自第三方模型，公开发布前应确认其许可证和再分发权限。

## 实验数据

实验原始 CSV 和结果图位于 [experiments/](experiments/)，对应的来源和文件统计见 [experiments/manifest.yaml](experiments/manifest.yaml)。

绘图脚本位于 `scripts/rsl_rl/`，可以根据新的评估 CSV 重新生成图表。仓库中的图像是已保存的实验结果，不是运行时依赖。

## 配置和代码位置

- 环境与奖励：`source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/wheel_leg_env_cfg.py`；
- PPO 配置：`source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/agents/rsl_rl_ppo_cfg.py`；
- 任务注册：`source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/__init__.py`；
- 训练脚本：`scripts/rsl_rl/train.py`；
- 评估脚本：`scripts/rsl_rl/eval_*.py`；
- 配置索引：`configs/task_catalog.yaml`。

## 代码格式检查

```bash
pip install pre-commit
pre-commit run --all-files
```

## 隐私和公开范围

仓库不应包含个人姓名、学号、桌面路径、云桌面用户名、论文归档材料、答辩材料或本地环境配置。公开前请再次检查新增日志、配置快照、模型元数据和压缩包。

## 许可证

本仓库当前仍需要根据代码、CAD、USD、模型和第三方资产的来源分别确认许可证。正式公开前请补充根目录 `LICENSE`，并在资产目录中说明不同文件的许可范围。

# Wheel-leg simulation assets

This directory contains the robot assets required by the Isaac Lab simulation.

- `urdf/wheel_leg_correct/`: corrected SW2URDF export, including STL meshes and ROS metadata.
- `usd/wheel_leg_correct/`: USD export and its referenced USD configuration files.

The USD asset is loaded by `source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/wheel_leg_env_cfg.py`.

## 中文说明

这里保存轮腿机器人仿真所需的 URDF、STL 和 USD 资产。`urdf/wheel_leg_correct/` 是修正版 SW2URDF 导出结果，`usd/wheel_leg_correct/` 是 Isaac Lab 使用的 USD 及其依赖文件。

仿真配置会从仓库路径加载 USD，不依赖个人电脑、云桌面或其他本地绝对路径。使用这些资产时请保持目录结构不变。

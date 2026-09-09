# Wheel-leg simulation assets

This directory contains the robot assets required by the Isaac Lab simulation.

- `urdf/wheel_leg_correct/`: corrected SW2URDF export, including STL meshes and ROS metadata.
- `usd/wheel_leg_correct/`: USD export and its referenced USD configuration files.

The USD asset is loaded by `source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/wheel_leg_env_cfg.py`.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Debug Script: Send Zero Actions to the Environment
用于调试物理穿模或参数爆炸的问题。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# 1. 设置命令行参数
parser = argparse.ArgumentParser(description="Debug script with zero actions.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (e.g. Template-Wheel-Leg-v0).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate (default: 4 for easy viewing).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Omniverse 应用 (必须在导入物理模块之前)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. 导入 IsaacLab 和相关环境
import gymnasium as gym
import torch
import time

# 导入配置解析工具
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# 导入你的任务注册文件
import wheel_leg.tasks  # noqa: F401

def main():
    print(f"[INFO] Starting zero-action debug for task: {args_cli.task}")
    
    # 🚨 核心修复：解析任务的默认配置
    env_cfg = parse_env_cfg(args_cli.task)
    
    # 覆盖并行环境数量为较小的值，方便肉眼观察
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 🚨 核心修复：将 cfg 传入 gym.make
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 打印环境信息
    num_envs = env.unwrapped.num_envs
    num_actions = env.unwrapped.action_manager.total_action_dim
    device = env.unwrapped.device
    print(f"[INFO] Number of environments: {num_envs}")
    print(f"[INFO] Action space dimension: {num_actions}")
    print(f"[INFO] Using device: {device}")

    # 环境复位
    env.reset()
    
    # 生成全 0 动作张量
    zero_actions = torch.zeros((num_envs, num_actions), dtype=torch.float32, device=device)

    dt = env.unwrapped.step_dt
    print("[INFO] Entering simulation loop with ZERO actions...")
    
    # 开始仿真循环
    while simulation_app.is_running():
        start_time = time.time()
        
        # 将全 0 动作发送给环境
        obs, rewards, terminated, truncated, info = env.step(zero_actions)
        
        # 保持真实的观看速度
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
    simulation_app.close()

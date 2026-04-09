import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import time
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import wheel_leg.tasks

def main():
    env_cfg = parse_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    num_envs = env.unwrapped.num_envs
    num_actions = env.unwrapped.action_manager.total_action_dim
    device = env.unwrapped.device
    
    env.reset()
    print("[INFO] Entering simulation loop with RANDOM actions (Simulating PPO)...")
    
    step_count = 0
    while simulation_app.is_running():
        # 🚨 生成 [-1.0, 1.0] 的随机动作，模拟初期的神经网络
        random_actions = 2.0 * torch.rand((num_envs, num_actions), device=device) - 1.0
        
        obs, rewards, terminated, truncated, info = env.step(random_actions)
        step_count += 1
        
        # 🚨 NaN 探测器！
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        if torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any():
            print(f"\n💥💥💥 FATAL at step {step_count}: 观察值(Observation)出现了 NaN 💥💥💥")
            print("这意味着物理引擎被这些随机动作干碎了！")
            break
            
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"\n💥💥💥 FATAL at step {step_count}: 奖励(Rewards)出现了 NaN 💥💥💥")
            break
            
        time.sleep(env.unwrapped.step_dt)

if __name__ == "__main__":
    main()
    simulation_app.close()

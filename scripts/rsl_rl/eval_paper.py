# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import csv
import os
import sys
from pathlib import Path
import math

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate terrain traversal for paper results.")
parser.add_argument("--task", type=str, required=True, help="Name of the eval task (e.g., Eval-Stairs-v0).")
parser.add_argument("--eval-mode", type=str, choices=["stairs", "rough"], required=True, help="Which experiment logic to run.")
parser.add_argument("--command-vy", type=float, default=1.0, help="Constant base velocity command along local y.")
parser.add_argument("--target-dist", type=float, default=3.5, help="Distance (meters) considered as success for stairs.")
parser.add_argument("--record-steps", type=int, default=1000, help="Max steps for window (rough) or timeout (stairs).")
parser.add_argument("--num-trials", type=int, default=20, help="Number of repeated traversal trials.")
parser.add_argument("--output-prefix", type=str, default=None, help="Prefix for exported CSV.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat

import wheel_leg.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

def resolve_output_prefix(run_dir: str) -> Path:
    if args_cli.output_prefix:
        return Path(args_cli.output_prefix).expanduser().resolve()
    out_dir = Path(run_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_stub = f"eval_{args_cli.eval_mode}_vy{args_cli.command_vy:+.2f}"
    return out_dir / file_stub

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    # 强制只用单个环境按顺序测试，结果最准确
    env_cfg.scene.num_envs = 1 
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    run_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    command_buf = env.unwrapped.command_manager.get_command("base_velocity")
    
    trial_rows =[]

    for trial_idx in range(args_cli.num_trials):
        print(f"[INFO] Evaluating Trial {trial_idx + 1}/{args_cli.num_trials} [{args_cli.eval_mode} mode]")
        obs, _ = env.reset()
        
        # Settle steps (预热)
        for _ in range(50):
            command_buf[:, :3] = torch.tensor([[0.0, 0.0, 0.0]], dtype=command_buf.dtype, device=command_buf.device)
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))

        start_root_pos_y = float(robot.data.root_pos_w[0, 1].item())
        
        # 用于记录状态的变量
        terminated_step = None
        passed_target = False
        pitch_history = []
        roll_history =[]

        for step_idx in range(args_cli.record_steps):
            # 持续下发指令
            command_buf[:, :3] = torch.tensor(
                [[0.0, args_cli.command_vy, 0.0]], dtype=command_buf.dtype, device=command_buf.device
            )
            with torch.no_grad():
                actions = policy(obs)
                obs, _, terminated, _ = env.step(actions)
            
            # 获取当前状态
            current_y = float(robot.data.root_pos_w[0, 1].item())
            distance_y = current_y - start_root_pos_y
            
            # 记录姿态角用于平稳性分析
            quat_w = robot.data.root_quat_w[0:1]
            roll_rad, pitch_rad, _ = euler_xyz_from_quat(quat_w)
            pitch_history.append(float(torch.rad2deg(pitch_rad)[0].item()))
            roll_history.append(float(torch.rad2deg(roll_rad)[0].item()))

            # --- 模式 1：台阶测试逻辑 (Stairs) ---
            if args_cli.eval_mode == "stairs":
                if distance_y >= args_cli.target_dist:
                    passed_target = True
                    terminated_step = step_idx
                    break # 成功过关，提前结束
                if terminated[0].item():
                    terminated_step = step_idx
                    passed_target = False # 摔倒或发生限制
                    break

            # --- 模式 2：崎岖路面逻辑 (Rough) ---
            if args_cli.eval_mode == "rough":
                if terminated[0].item():
                    terminated_step = step_idx
                    break # 记录跌倒时刻

        # 结算当前 Trial 的数据
        end_root_pos = robot.data.root_pos_w[0]
        final_distance_y = float(end_root_pos[1].item() - start_root_pos_y)
        
        pitch_arr = np.array(pitch_history)
        roll_arr = np.array(roll_history)

        if args_cli.eval_mode == "stairs":
            # 台阶指标：是否通过，通过耗时
            success = 1.0 if passed_target else 0.0
            time_taken_s = (terminated_step * env_cfg.sim.dt * env_cfg.decimation) if passed_target else -1.0
            trial_rows.append({
                "trial_idx": trial_idx,
                "success": success,
                "time_taken_s": time_taken_s,
                "final_distance_y": final_distance_y,
                "failed_at_step": terminated_step if not passed_target else -1,
            })

        elif args_cli.eval_mode == "rough":
            # 崎岖指标：平均距离，终止率，姿态方差(稳定性)
            survived = 1.0 if terminated_step is None else 0.0
            trial_rows.append({
                "trial_idx": trial_idx,
                "survived": survived,
                "distance_y": final_distance_y,
                "pitch_std_deg": float(np.std(pitch_arr)),
                "roll_std_deg": float(np.std(roll_arr)),
                "mean_abs_pitch_deg": float(np.mean(np.abs(pitch_arr))),
            })

    # --- 导出 CSV 数据 ---
    output_prefix = resolve_output_prefix(run_dir)
    trials_csv = output_prefix.with_name(output_prefix.name + "_trials.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")

    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)

    if args_cli.eval_mode == "stairs":
        success_rate = sum(r["success"] for r in trial_rows) / len(trial_rows)
        valid_times = [r["time_taken_s"] for r in trial_rows if r["success"] == 1.0]
        avg_time = sum(valid_times)/len(valid_times) if valid_times else -1.0
        summary = {
            "eval_mode": "stairs",
            "success_rate": success_rate,
            "avg_time_to_cross_s": avg_time,
            "target_distance": args_cli.target_dist
        }
    else:
        survival_rate = sum(r["survived"] for r in trial_rows) / len(trial_rows)
        summary = {
            "eval_mode": "rough",
            "survival_rate": survival_rate,
            "mean_forward_distance": sum(r["distance_y"] for r in trial_rows) / len(trial_rows),
            "mean_pitch_std_deg": sum(r["pitch_std_deg"] for r in trial_rows) / len(trial_rows),
            "mean_roll_std_deg": sum(r["roll_std_deg"] for r in trial_rows) / len(trial_rows),
        }

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    env.close()
    print(f"[INFO] Traversal summary exported to: {summary_csv}")

if __name__ == "__main__":
    main()
    simulation_app.close()
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
    timeseries_rows = []

    for trial_idx in range(args_cli.num_trials):
        print(f"[INFO] Evaluating Trial {trial_idx + 1}/{args_cli.num_trials} [{args_cli.eval_mode} mode]")
        obs, _ = env.reset()

        # --- 强制改变初始位置，让小车从楼梯底端开始 ---
        env_ids = torch.tensor([0], device=env.unwrapped.device, dtype=torch.long)
        root_pos_w = robot.data.root_pos_w[0:1].clone()
        root_quat_w = robot.data.root_quat_w[0:1].clone()
        # 把小车往 Y 轴负方向挪 2 米，这样它面前就是向上的台阶
        root_pos_w[:, 1] -= 2.0  
        root_pose = torch.cat([root_pos_w, root_quat_w], dim=-1)
        root_vel = torch.zeros((1, 6), device=env.unwrapped.device)
        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        robot.reset(env_ids)
        
        # 重置位置后，必须刷新一次 obs！否则第一步接收到的是错误的传感器数据
        obs, _ = env.get_observations()
        
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

        # 【新增】：用于保存 env.step() 之前的真实物理状态，防止触发 reset 后数据丢失
        last_distance_y = 0.0
        last_pitch_deg = 0.0
        last_roll_deg = 0.0
        last_z_height = 0.0

        for step_idx in range(args_cli.record_steps):
            # --- 1. 获取动作【前】的真实状态（此时绝不会是 reset 后的虚假状态） ---
            current_y = float(robot.data.root_pos_w[0, 1].item())
            current_z = float(robot.data.root_pos_w[0, 2].item())
            distance_y = current_y - start_root_pos_y
            
            quat_w = robot.data.root_quat_w[0:1]
            roll_rad, pitch_rad, _ = euler_xyz_from_quat(quat_w)
            roll_deg = float(torch.rad2deg(roll_rad)[0].item())
            pitch_deg = float(torch.rad2deg(pitch_rad)[0].item())

            # 备份到 last 变量
            last_distance_y = distance_y
            last_pitch_deg = pitch_deg
            last_roll_deg = roll_deg
            last_z_height = current_z
            
            pitch_history.append(pitch_deg)
            roll_history.append(roll_deg)

            # --- 2. 记录时序数据 (仅限于 Rough 第一轮) ---
            if args_cli.eval_mode == "rough" and trial_idx == 0:
                current_time = step_idx * env_cfg.sim.dt * env_cfg.decimation
                step_data = {
                    "time_s": current_time,
                    "roll_deg": roll_deg,
                    "pitch_deg": pitch_deg,
                    "distance_y": distance_y,
                }
                for j_idx, j_name in enumerate(robot.data.joint_names):
                    step_data[f"joint_{j_name}"] = float(robot.data.joint_pos[0, j_idx].item())
                timeseries_rows.append(step_data)

            # --- 3. 持续下发指令并执行 Step ---
            command_buf[:, :3] = torch.tensor(
                [[0.0, args_cli.command_vy, 0.0]], dtype=command_buf.dtype, device=command_buf.device
            )
            with torch.no_grad():
                actions = policy(obs)
                # 【注意】：必须接收 extras 字典，里面包含了 timeout 等内部信息
                obs, _, terminated, extras = env.step(actions)
            
            # ==========================================
            # 内部函数：基于摔倒前的一瞬间状态，诊断死因
            # ==========================================
            def get_termination_reason():
                if "time_outs" in extras and extras["time_outs"][0].item():
                    return "⏱️ Time Out (Reached Max Steps)"
                # 如果 Pitch/Roll 大于 ~70度，通常是翻车
                if abs(last_pitch_deg) > 70.0 or abs(last_roll_deg) > 70.0:
                    return f"🤸 Bad Orientation (Pitch={last_pitch_deg:.1f}°, Roll={last_roll_deg:.1f}°)"
                # 如果 Z 高度极低，通常是腿软底盘砸地
                elif last_z_height < 0.12:  
                    return f"💥 Base Crash / Height Too Low (Z={last_z_height:.3f}m)"
                else:
                    return f"⚠️ Joint Limits / Other (Pos or Vel exceeded limits)"

            # --- 模式 1：台阶测试逻辑 (Stairs) ---
            if args_cli.eval_mode == "stairs":
                if distance_y >= args_cli.target_dist:
                    passed_target = True
                    terminated_step = step_idx
                    print(f"      ✅ [Success] Crossed target distance {args_cli.target_dist}m at step {step_idx}.")
                    break 
                if terminated[0].item():
                    terminated_step = step_idx
                    passed_target = False 
                    reason = get_termination_reason()
                    print(f"      ❌ [Failed] Terminated at step {step_idx}. Reason: {reason}")
                    break

            # --- 模式 2：崎岖路面逻辑 (Rough) ---
            if args_cli.eval_mode == "rough":
                if terminated[0].item():
                    terminated_step = step_idx
                    reason = get_termination_reason()
                    print(f"      ❌ [Failed] Terminated at step {step_idx}. Reason: {reason}")
                    break 
        
        # 正常跑完全程没有摔倒（触发 Time Out）
        if args_cli.eval_mode == "rough" and terminated_step is None:
            print(f"      ✅ [Survived] Completed all {args_cli.record_steps} steps flawlessly.")

        # 结算当前 Trial 的数据
        # 【核心修正】：使用 last_distance_y 而不是 robot.data，确保获取的是摔倒前跑出的真实距离！
        final_distance_y = last_distance_y
        
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

    # 【新增：保存时序数据】
    if args_cli.eval_mode == "rough" and len(timeseries_rows) > 0:
        timeseries_csv = output_prefix.with_name(output_prefix.name + "_timeseries.csv")
        with timeseries_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(timeseries_rows[0].keys()))
            writer.writeheader()
            writer.writerows(timeseries_rows)
        print(f"[INFO] Traversal timeseries (Trial 0) exported to: {timeseries_csv}")

    env.close()
    print(f"[INFO] Traversal summary exported to: {summary_csv}")

if __name__ == "__main__":
    main()
    simulation_app.close()
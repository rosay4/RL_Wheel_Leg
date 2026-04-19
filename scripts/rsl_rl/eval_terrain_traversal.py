# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate discrete-step / rough / unstructured terrain traversal performance."""

import argparse
import csv
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate terrain traversal success rate of an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--command-vy", type=float, default=1.0, help="Constant base velocity command along local y.")
parser.add_argument("--command-wz", type=float, default=0.0, help="Constant yaw-rate command.")
parser.add_argument("--settle-steps", type=int, default=100, help="Warm-up steps before each traversal trial.")
parser.add_argument("--record-steps", type=int, default=500, help="Maximum number of steps per traversal trial.")
parser.add_argument("--num-trials", type=int, default=20, help="Number of repeated traversal trials.")
parser.add_argument(
    "--spawn-shift-y",
    type=float,
    default=0.0,
    help="Optional world-y shift after reset, useful when the interesting terrain is away from the center platform.",
)
parser.add_argument(
    "--spawn-shift-z",
    type=float,
    default=0.0,
    help="Optional extra world-z lift after reset.",
)
parser.add_argument(
    "--output-prefix",
    type=str,
    default=None,
    help="Optional file prefix for exported CSV files. Defaults inside the run folder.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import wheel_leg.tasks  # noqa: F401


def _resolve_output_prefix(run_dir: str) -> Path:
    if args_cli.output_prefix:
        return Path(args_cli.output_prefix).expanduser().resolve()
    out_dir = Path(run_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_stub = f"terrain_traversal_vy_{args_cli.command_vy:+.2f}_wz_{args_cli.command_wz:+.2f}"
    return out_dir / file_stub


def _optional_spawn_shift(robot, device: torch.device):
    if abs(args_cli.spawn_shift_y) < 1.0e-9 and abs(args_cli.spawn_shift_z) < 1.0e-9:
        return
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    root_pos_w = robot.data.root_pos_w[0:1].clone()
    root_quat_w = robot.data.root_quat_w[0:1].clone()
    root_pos_w[:, 1] += args_cli.spawn_shift_y
    root_pos_w[:, 2] += args_cli.spawn_shift_z
    root_pose = torch.cat([root_pos_w, root_quat_w], dim=-1)
    root_vel = torch.zeros((1, 6), device=device)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    robot.reset(env_ids)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
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
    trial_rows: list[dict[str, float]] = []

    for trial_idx in range(args_cli.num_trials):
        print(f"[INFO] Traversal trial {trial_idx + 1}/{args_cli.num_trials}")
        obs, _ = env.reset()
        _optional_spawn_shift(robot, env.unwrapped.device)
        obs, _ = env.get_observations()
        for _ in range(args_cli.settle_steps):
            command_buf[:, :3] = torch.tensor(
                [[0.0, args_cli.command_vy, args_cli.command_wz]], dtype=command_buf.dtype, device=command_buf.device
            )
            with torch.no_grad():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

        start_root_pos_y = float(robot.data.root_pos_w[0, 1].item())
        terminated_step = None
        final_pitch_deg = None
        for step_idx in range(args_cli.record_steps):
            command_buf[:, :3] = torch.tensor(
                [[0.0, args_cli.command_vy, args_cli.command_wz]], dtype=command_buf.dtype, device=command_buf.device
            )
            with torch.no_grad():
                actions = policy(obs)
                obs, _, terminated, _ = env.step(actions)
            if terminated[0].item():
                terminated_step = step_idx
                break

        end_root_pos = robot.data.root_pos_w[0]
        quat_w = robot.data.root_quat_w[0:1]
        from isaaclab.utils.math import euler_xyz_from_quat  # local import to keep startup order simple

        _, pitch_rad, _ = euler_xyz_from_quat(quat_w)
        final_pitch_deg = float(torch.rad2deg(pitch_rad)[0].item())

        success = 1.0 if terminated_step is None else 0.0
        trial_rows.append(
            {
                "trial_idx": float(trial_idx),
                "command_vy": float(args_cli.command_vy),
                "command_wz": float(args_cli.command_wz),
                "success": success,
                "terminated_step": float(terminated_step if terminated_step is not None else args_cli.record_steps),
                "distance_y_w": float(end_root_pos[1].item() - start_root_pos_y),
                "final_root_z_w": float(end_root_pos[2].item()),
                "final_pitch_deg": final_pitch_deg,
            }
        )

    output_prefix = _resolve_output_prefix(run_dir)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trials_csv = output_prefix.with_name(output_prefix.name + "_trials.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")

    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)

    summary = {
        "command_vy": float(args_cli.command_vy),
        "command_wz": float(args_cli.command_wz),
        "num_trials": float(args_cli.num_trials),
        "success_rate": sum(row["success"] for row in trial_rows) / len(trial_rows),
        "mean_distance_y_w": sum(row["distance_y_w"] for row in trial_rows) / len(trial_rows),
        "mean_terminated_step": sum(row["terminated_step"] for row in trial_rows) / len(trial_rows),
        "mean_final_pitch_deg": sum(row["final_pitch_deg"] for row in trial_rows) / len(trial_rows),
    }
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    env.close()
    print(f"[INFO] Traversal trial details exported to: {trials_csv}")
    print(f"[INFO] Traversal summary exported to: {summary_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()

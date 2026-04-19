# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate slope locomotion performance and export pose / COM-height trajectories."""

import argparse
import csv
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate slope locomotion performance of an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--command-vy", type=float, default=1.0, help="Constant base velocity command along local y.")
parser.add_argument("--command-wz", type=float, default=0.0, help="Constant yaw-rate command.")
parser.add_argument("--settle-steps", type=int, default=120, help="Warm-up steps before recording.")
parser.add_argument("--record-steps", type=int, default=360, help="Number of recorded steps.")
parser.add_argument(
    "--snapshot-steps",
    type=int,
    nargs="+",
    default=[0, 60, 120, 180, 240, 300, 359],
    help="Recorded step indices that should also be exported as pose snapshots.",
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
from isaaclab.utils.math import euler_xyz_from_quat
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
    file_stub = f"slope_eval_vy_{args_cli.command_vy:+.2f}_wz_{args_cli.command_wz:+.2f}"
    return out_dir / file_stub


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
    dt = env.unwrapped.step_dt

    obs, _ = env.reset()
    for _ in range(args_cli.settle_steps):
        command_buf[:, :3] = torch.tensor(
            [[0.0, args_cli.command_vy, args_cli.command_wz]], dtype=command_buf.dtype, device=command_buf.device
        )
        with torch.no_grad():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    rows: list[dict[str, float]] = []
    snapshot_rows: list[dict[str, float]] = []
    for step_idx in range(args_cli.record_steps):
        command_buf[:, :3] = torch.tensor(
            [[0.0, args_cli.command_vy, args_cli.command_wz]], dtype=command_buf.dtype, device=command_buf.device
        )
        with torch.no_grad():
            actions = policy(obs)
            obs, _, terminated, _ = env.step(actions)

        quat_w = robot.data.root_quat_w[0:1]
        roll_rad, pitch_rad, yaw_rad = euler_xyz_from_quat(quat_w)
        root_pos_w = robot.data.root_pos_w[0]
        root_lin_vel_b = getattr(robot.data, "root_lin_vel_b", None)
        if root_lin_vel_b is None:
            root_lin_vel_b = getattr(robot.data, "root_com_lin_vel_b")
        root_ang_vel_b = getattr(robot.data, "root_ang_vel_b", None)
        if root_ang_vel_b is None:
            root_ang_vel_b = getattr(robot.data, "root_com_ang_vel_b")
        com_pos_w = getattr(robot.data, "root_com_pos_w", None)
        com_height = float(com_pos_w[0, 2].item()) if com_pos_w is not None else float(root_pos_w[2].item())

        row = {
            "time_s": step_idx * dt,
            "step": step_idx,
            "command_vy": float(args_cli.command_vy),
            "command_wz": float(args_cli.command_wz),
            "root_x_w": float(root_pos_w[0].item()),
            "root_y_w": float(root_pos_w[1].item()),
            "root_z_w": float(root_pos_w[2].item()),
            "com_height_w": com_height,
            "roll_deg": float(torch.rad2deg(roll_rad)[0].item()),
            "pitch_deg": float(torch.rad2deg(pitch_rad)[0].item()),
            "yaw_deg": float(torch.rad2deg(yaw_rad)[0].item()),
            "lin_vel_y_b": float(root_lin_vel_b[0, 1].item()),
            "ang_vel_z_b": float(root_ang_vel_b[0, 2].item()),
            "terminated": float(terminated[0].item()),
        }
        rows.append(row)
        if step_idx in args_cli.snapshot_steps:
            snapshot_rows.append(row.copy())
        if terminated[0].item():
            break

    output_prefix = _resolve_output_prefix(run_dir)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trajectory_csv = output_prefix.with_name(output_prefix.name + "_trajectory.csv")
    snapshot_csv = output_prefix.with_name(output_prefix.name + "_snapshots.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")

    with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with snapshot_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(snapshot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(snapshot_rows)

    summary = {
        "command_vy": float(args_cli.command_vy),
        "command_wz": float(args_cli.command_wz),
        "duration_s": float(rows[-1]["time_s"]),
        "distance_y_w": float(rows[-1]["root_y_w"] - rows[0]["root_y_w"]),
        "mean_pitch_deg": sum(row["pitch_deg"] for row in rows) / len(rows),
        "max_abs_pitch_deg": max(abs(row["pitch_deg"]) for row in rows),
        "mean_com_height_w": sum(row["com_height_w"] for row in rows) / len(rows),
        "min_com_height_w": min(row["com_height_w"] for row in rows),
        "terminated": rows[-1]["terminated"],
    }
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    env.close()
    print(f"[INFO] Slope trajectory exported to: {trajectory_csv}")
    print(f"[INFO] Slope snapshots exported to: {snapshot_csv}")
    print(f"[INFO] Slope summary exported to: {summary_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()

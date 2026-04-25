# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate wheel torque switching behavior for reward-ablation policies."""

import argparse
import csv
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate wheel-hub torque switching behavior of an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Wheel-Leg-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--model-name", type=str, default="policy", help="Model label written into exported CSV rows.")
parser.add_argument("--command-vy", type=float, default=1.0, help="Constant base velocity command along local y.")
parser.add_argument("--command-wz", type=float, default=0.0, help="Constant yaw-rate command.")
parser.add_argument("--settle-steps", type=int, default=120, help="Warm-up steps before torque recording.")
parser.add_argument("--record-steps", type=int, default=500, help="Number of steps used to record torque response.")
parser.add_argument(
    "--output-prefix",
    type=str,
    default=None,
    help="Optional prefix for exported CSV files. Defaults inside the run folder.",
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
    file_stub = f"torque_bangbang_{args_cli.model_name}_vy_{args_cli.command_vy:+.2f}_wz_{args_cli.command_wz:+.2f}"
    return out_dir / file_stub


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
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
    wheel_joint_ids, wheel_joint_names = robot.find_joints(".*ankle.*", preserve_order=True)
    if len(wheel_joint_ids) == 0:
        raise RuntimeError("No wheel joints matched '.*ankle.*'; cannot evaluate wheel torque behavior.")

    output_prefix = _resolve_output_prefix(run_dir)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    timeseries_csv = output_prefix.with_name(output_prefix.name + "_timeseries.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    print(f"[INFO] Exporting torque timeseries to: {timeseries_csv}")
    print(f"[INFO] Exporting torque summary to: {summary_csv}")

    obs, _ = env.reset()
    command_buf = env.unwrapped.command_manager.get_command("base_velocity")
    dt = env.unwrapped.step_dt

    for _ in range(args_cli.settle_steps):
        command_buf[:, :3] = torch.tensor(
            [[0.0, args_cli.command_vy, args_cli.command_wz]],
            dtype=command_buf.dtype,
            device=command_buf.device,
        )
        with torch.no_grad():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    rows: list[dict[str, float | str]] = []
    torque_samples = []
    action_samples = []
    for step_idx in range(args_cli.record_steps):
        command_buf[:, :3] = torch.tensor(
            [[0.0, args_cli.command_vy, args_cli.command_wz]],
            dtype=command_buf.dtype,
            device=command_buf.device,
        )
        with torch.no_grad():
            actions = policy(obs)
            obs, _, terminated, _ = env.step(actions)

        wheel_torque = robot.data.applied_torque[0, wheel_joint_ids].detach().clone()
        wheel_actions = actions[0].detach().clone()
        torque_samples.append(wheel_torque)
        action_samples.append(wheel_actions)

        row = {
            "model_name": args_cli.model_name,
            "time_s": step_idx * dt,
            "command_vy": float(args_cli.command_vy),
            "command_wz": float(args_cli.command_wz),
            "terminated": float(terminated[0].item()),
        }
        for joint_name, torque_value in zip(wheel_joint_names, wheel_torque, strict=True):
            row[f"torque_{joint_name}"] = float(torque_value.item())
        rows.append(row)

        if terminated[0].item():
            break

    torque_tensor = torch.stack(torque_samples, dim=0)
    torque_diff = torque_tensor[1:] - torque_tensor[:-1] if torque_tensor.shape[0] > 1 else torch.zeros_like(torque_tensor)
    abs_torque = torch.abs(torque_tensor)
    sign_changes = torch.sum(torch.sign(torque_tensor[1:]) != torch.sign(torque_tensor[:-1]), dim=0) if torque_tensor.shape[0] > 1 else torch.zeros(torque_tensor.shape[1])

    summary = {
        "model_name": args_cli.model_name,
        "command_vy": float(args_cli.command_vy),
        "command_wz": float(args_cli.command_wz),
        "recorded_steps": float(torque_tensor.shape[0]),
        "mean_abs_torque": float(torch.mean(abs_torque).item()),
        "max_abs_torque": float(torch.max(abs_torque).item()),
        "torque_std": float(torch.std(torque_tensor).item()),
        "torque_diff_rms": float(torch.sqrt(torch.mean(torque_diff**2)).item()),
        "mean_sign_changes_per_joint": float(torch.mean(sign_changes.float()).item()),
        "terminated": float(rows[-1]["terminated"]),
    }
    for joint_idx, joint_name in enumerate(wheel_joint_names):
        joint_torque = torque_tensor[:, joint_idx]
        joint_diff = torque_diff[:, joint_idx] if torque_diff.numel() > 0 else torch.zeros_like(joint_torque)
        summary[f"mean_abs_torque_{joint_name}"] = float(torch.mean(torch.abs(joint_torque)).item())
        summary[f"torque_diff_rms_{joint_name}"] = float(torch.sqrt(torch.mean(joint_diff**2)).item())
        summary[f"sign_changes_{joint_name}"] = float(sign_changes[joint_idx].item())

    with timeseries_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    env.close()
    print("[INFO] Torque switching evaluation finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()

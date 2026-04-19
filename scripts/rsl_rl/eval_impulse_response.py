# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate transient impulse response and export pitch / wheel torque curves."""

import argparse
import csv
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate transient impulse response of an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--command-vy", type=float, default=0.0, help="Constant base velocity command along local y.")
parser.add_argument("--command-wz", type=float, default=0.0, help="Constant yaw-rate command.")
parser.add_argument("--settle-steps", type=int, default=150, help="Warm-up steps before the impulse is applied.")
parser.add_argument("--eval-steps", type=int, default=600, help="Total evaluation steps to record.")
parser.add_argument("--impulse-start-step", type=int, default=200, help="Step index at which the impulse starts.")
parser.add_argument("--impulse-duration-steps", type=int, default=8, help="Number of steps for the impulse pulse.")
parser.add_argument(
    "--impulse-torque-pitch",
    type=float,
    default=45.0,
    help="Pitch-axis external torque magnitude in N*m applied to the base.",
)
parser.add_argument(
    "--impulse-frame",
    type=str,
    choices=("global", "local"),
    default="global",
    help="Frame used for the external impulse torque.",
)
parser.add_argument(
    "--output-csv",
    type=str,
    default=None,
    help="Optional absolute/relative path for the exported CSV. Defaults inside the run folder.",
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


def _resolve_output_csv(default_root: str, run_dir: str) -> Path:
    if args_cli.output_csv:
        return Path(args_cli.output_csv).expanduser().resolve()
    out_dir = Path(run_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"impulse_response_vy_{args_cli.command_vy:+.2f}_wz_{args_cli.command_wz:+.2f}"
        f"_tp_{args_cli.impulse_torque_pitch:.1f}.csv"
    )
    return out_dir / file_name


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run a pitch-impulse response evaluation and dump a CSV."""
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
    wheel_joint_ids, wheel_joint_names = robot.find_joints(".*ankle.*", preserve_order=True)
    if len(wheel_joint_ids) == 0:
        raise RuntimeError("No wheel joints matched '.*ankle.*'; cannot export hub torque response.")
    base_body_ids, base_body_names = robot.find_bodies(".*base.*", preserve_order=True)
    if len(base_body_ids) == 0:
        base_body_ids = [0]
        base_body_names = [robot.body_names[0]]

    base_body_id = base_body_ids[0]
    env_ids = torch.tensor([0], device=env.unwrapped.device, dtype=torch.long)
    body_ids = torch.tensor([base_body_id], device=env.unwrapped.device, dtype=torch.long)

    output_csv = _resolve_output_csv(log_root_path, run_dir)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    print(f"[INFO] Applying impulse to body: {base_body_names[0]}")
    print(f"[INFO] Exporting CSV to: {output_csv}")

    obs, _ = env.reset()
    command_buf = env.unwrapped.command_manager.get_command("base_velocity")
    dt = env.unwrapped.step_dt

    # warm-up before recording
    zero_forces = torch.zeros((1, 1, 3), device=env.unwrapped.device)
    zero_torques = torch.zeros((1, 1, 3), device=env.unwrapped.device)
    for _ in range(args_cli.settle_steps):
        command_buf[:, :3] = torch.tensor(
            [[0.0, args_cli.command_vy, args_cli.command_wz]],
            dtype=command_buf.dtype,
            device=command_buf.device,
        )
        robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=zero_torques,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=args_cli.impulse_frame == "global",
        )
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    rows: list[dict[str, float]] = []
    for step_idx in range(args_cli.eval_steps):
        command_buf[:, :3] = torch.tensor(
            [[0.0, args_cli.command_vy, args_cli.command_wz]],
            dtype=command_buf.dtype,
            device=command_buf.device,
        )

        apply_impulse = args_cli.impulse_start_step <= step_idx < (
            args_cli.impulse_start_step + args_cli.impulse_duration_steps
        )
        torque_value = args_cli.impulse_torque_pitch if apply_impulse else 0.0
        impulse_torque = torch.tensor([[[0.0, torque_value, 0.0]]], device=env.unwrapped.device)
        robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=impulse_torque,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=args_cli.impulse_frame == "global",
        )

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        quat_w = robot.data.root_quat_w[0:1]
        _, pitch_rad, _ = euler_xyz_from_quat(quat_w)
        applied_torque = robot.data.applied_torque[0]
        row = {
            "time_s": step_idx * dt,
            "pitch_rad": float(pitch_rad[0].item()),
            "pitch_deg": float(torch.rad2deg(pitch_rad)[0].item()),
            "pitch_rate_rad_s": float(robot.data.root_ang_vel_b[0, 1].item()),
            "command_vy": float(args_cli.command_vy),
            "command_wz": float(args_cli.command_wz),
            "impulse_torque_pitch": float(torque_value),
        }
        for joint_id, joint_name in zip(wheel_joint_ids, wheel_joint_names, strict=True):
            row[f"torque_{joint_name}"] = float(applied_torque[joint_id].item())
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    robot.set_external_force_and_torque(
        forces=zero_forces,
        torques=zero_torques,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=args_cli.impulse_frame == "global",
    )
    env.close()
    print("[INFO] Impulse-response evaluation finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()

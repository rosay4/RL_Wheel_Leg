# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate velocity tracking and turning performance for a trained policy."""

import argparse
import csv
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate velocity / turning tracking performance of an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--vy-list",
    type=float,
    nargs="+",
    default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    help="List of local-y velocity commands to evaluate.",
)
parser.add_argument(
    "--wz-list",
    type=float,
    nargs="+",
    default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    help="List of yaw-rate commands to evaluate.",
)
parser.add_argument("--settle-steps", type=int, default=120, help="Warm-up steps before statistics are collected.")
parser.add_argument("--record-steps", type=int, default=240, help="Number of steps used to compute metrics.")
parser.add_argument(
    "--print-progress",
    action="store_true",
    default=False,
    help="Print progress for each evaluated command pair.",
)
parser.add_argument(
    "--output-csv",
    type=str,
    default=None,
    help="Optional absolute/relative path for the summary CSV. Defaults inside the run folder.",
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


def _resolve_output_csv(run_dir: str) -> Path:
    if args_cli.output_csv:
        return Path(args_cli.output_csv).expanduser().resolve()
    out_dir = Path(run_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "tracking_performance_summary.csv"


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
    rows: list[dict[str, float]] = []

    for vy_cmd in args_cli.vy_list:
        for wz_cmd in args_cli.wz_list:
            if args_cli.print_progress:
                print(f"[INFO] Evaluating command pair vy={vy_cmd:+.2f}, wz={wz_cmd:+.2f}")
            obs, _ = env.reset()
            # settle
            for _ in range(args_cli.settle_steps):
                command_buf[:, :3] = torch.tensor(
                    [[0.0, vy_cmd, wz_cmd]], dtype=command_buf.dtype, device=command_buf.device
                )
                with torch.no_grad():
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)

            vy_err_hist = []
            wz_err_hist = []
            vy_act_hist = []
            wz_act_hist = []
            for _ in range(args_cli.record_steps):
                command_buf[:, :3] = torch.tensor(
                    [[0.0, vy_cmd, wz_cmd]], dtype=command_buf.dtype, device=command_buf.device
                )
                with torch.no_grad():
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)

                root_lin_vel_b = getattr(robot.data, "root_lin_vel_b", None)
                if root_lin_vel_b is None:
                    root_lin_vel_b = getattr(robot.data, "root_com_lin_vel_b")
                root_ang_vel_b = getattr(robot.data, "root_ang_vel_b", None)
                if root_ang_vel_b is None:
                    root_ang_vel_b = getattr(robot.data, "root_com_ang_vel_b")

                vy_actual = float(root_lin_vel_b[0, 1].item())
                wz_actual = float(root_ang_vel_b[0, 2].item())
                vy_act_hist.append(vy_actual)
                wz_act_hist.append(wz_actual)
                vy_err_hist.append(vy_cmd - vy_actual)
                wz_err_hist.append(wz_cmd - wz_actual)

            vy_err = torch.tensor(vy_err_hist)
            wz_err = torch.tensor(wz_err_hist)
            rows.append(
                {
                    "command_vy": vy_cmd,
                    "command_wz": wz_cmd,
                    "mean_vy_actual": float(torch.tensor(vy_act_hist).mean().item()),
                    "mean_wz_actual": float(torch.tensor(wz_act_hist).mean().item()),
                    "rmse_vy": float(torch.sqrt(torch.mean(vy_err**2)).item()),
                    "rmse_wz": float(torch.sqrt(torch.mean(wz_err**2)).item()),
                    "steady_err_vy": float(vy_err[-50:].mean().item()),
                    "steady_err_wz": float(wz_err[-50:].mean().item()),
                    "peak_err_vy": float(torch.max(torch.abs(vy_err)).item()),
                    "peak_err_wz": float(torch.max(torch.abs(wz_err)).item()),
                }
            )

    output_csv = _resolve_output_csv(run_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    env.close()
    print(f"[INFO] Tracking evaluation summary exported to: {output_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()

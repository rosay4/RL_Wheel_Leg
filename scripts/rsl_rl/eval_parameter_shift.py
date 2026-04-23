# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate policy robustness under fixed parameter-shift settings."""

import argparse
import copy
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate policy robustness under fixed parameter shifts.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Wheel-Leg-NoDr-v0",
    help="Evaluation task. Defaults to the flat no-DR task for a clean parameter-shift benchmark.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--model-name", type=str, default="policy", help="Model label written into exported CSV rows.")
parser.add_argument("--command-vy", type=float, default=0.6, help="Constant base velocity command along local y.")
parser.add_argument("--command-wz", type=float, default=0.0, help="Constant yaw-rate command.")
parser.add_argument("--settle-steps", type=int, default=120, help="Warm-up steps before statistics are collected.")
parser.add_argument("--record-steps", type=int, default=400, help="Maximum number of steps per trial.")
parser.add_argument("--num-trials", type=int, default=20, help="Number of repeated rollout trials for each shift set.")
parser.add_argument(
    "--shift-set",
    type=str,
    nargs="+",
    default=None,
    choices=(
        "nominal",
        "low_friction",
        "high_friction",
        "mass_plus",
        "mass_minus",
        "actuator_weak",
        "actuator_strong",
    ),
    help="Optional explicit list of shift sets. If omitted, the built-in paper suite is used.",
)
parser.add_argument(
    "--output-csv",
    type=str,
    default=None,
    help="Optional absolute/relative path for the main per-trial CSV. Can be reused across multiple runs.",
)
parser.add_argument(
    "--append-output",
    action="store_true",
    default=False,
    help="Append results to an existing main CSV instead of overwriting it.",
)
parser.add_argument(
    "--export-phase-csv",
    action="store_true",
    default=False,
    help="Also export a separate phase-portrait CSV for the first trial of each shift set.",
)
parser.add_argument(
    "--phase-csv",
    type=str,
    default=None,
    help="Optional path for phase CSV. Used only when --export-phase-csv is set.",
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

import isaaclab.envs.mdp as mdp
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import wheel_leg.tasks  # noqa: F401


DEFAULT_SHIFT_SET = "nominal"


@dataclass(frozen=True)
class ShiftSpec:
    friction: tuple[float, float] | None = None
    mass_delta: float | None = None
    actuator_scale_mult: float = 1.0
    actuator_effort_mult: float = 1.0


SHIFT_SPECS: dict[str, ShiftSpec] = {
    "nominal": ShiftSpec(),
    "low_friction": ShiftSpec(friction=(0.15, 0.10)),
    "high_friction": ShiftSpec(friction=(1.50, 1.30)),
    "mass_plus": ShiftSpec(mass_delta=0.80),
    "mass_minus": ShiftSpec(mass_delta=-0.35),
    "actuator_weak": ShiftSpec(actuator_scale_mult=0.50, actuator_effort_mult=0.50),
    "actuator_strong": ShiftSpec(actuator_scale_mult=1.20, actuator_effort_mult=1.20),
}


def _resolve_output_csv(run_dir: str) -> Path:
    if args_cli.output_csv:
        return Path(args_cli.output_csv).expanduser().resolve()
    out_dir = Path(run_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.append_output:
        return out_dir / f"parameter_shift_{args_cli.model_name}.csv"
    shift_name = _get_shift_names()[0]
    return out_dir / f"parameter_shift_{args_cli.model_name}_{shift_name}.csv"


def _resolve_phase_csv(run_dir: str) -> Path:
    if args_cli.phase_csv:
        return Path(args_cli.phase_csv).expanduser().resolve()
    out_dir = Path(run_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    shift_name = _get_shift_names()[0]
    return out_dir / f"parameter_shift_phase_{args_cli.model_name}_{shift_name}.csv"


def _get_shift_names() -> list[str]:
    if args_cli.shift_set:
        return [args_cli.shift_set[0]]
    return [DEFAULT_SHIFT_SET]


def _build_shifted_env_cfg(base_env_cfg, shift_name: str):
    env_cfg = copy.deepcopy(base_env_cfg)
    spec = SHIFT_SPECS[shift_name]

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        if hasattr(env_cfg.observations.policy, "enable_corruption"):
            env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg, "events"):
        # Start from a clean deterministic evaluation environment.
        env_cfg.events.robot_physics_material = None
        env_cfg.events.add_base_mass = None
        env_cfg.events.base_push = None

        if spec.friction is not None:
            static_friction, dynamic_friction = spec.friction
            env_cfg.events.robot_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (static_friction, static_friction),
                    "dynamic_friction_range": (dynamic_friction, dynamic_friction),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            )

        if spec.mass_delta is not None:
            env_cfg.events.add_base_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
                    "mass_distribution_params": (spec.mass_delta, spec.mass_delta),
                    "operation": "add",
                },
            )

    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "wheel_joints"):
        env_cfg.actions.wheel_joints.scale *= spec.actuator_scale_mult

    wheel_actuator = None
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "robot"):
        actuators = getattr(env_cfg.scene.robot, "actuators", {})
        wheel_actuator = actuators.get("wheels", None)
    if wheel_actuator is not None and hasattr(wheel_actuator, "effort_limit"):
        wheel_actuator.effort_limit *= spec.actuator_effort_mult

    return env_cfg


def _termination_reason(env) -> str:
    active_terms = list(env.unwrapped.termination_manager.get_active_iterable_terms(0))
    active_names = [term_name for term_name, term_values in active_terms if any(float(v) > 0.0 for v in term_values)]
    return "|".join(active_names) if active_names else "unknown"


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(base_env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    base_env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    run_dir = os.path.dirname(resume_path)

    trial_rows: list[dict[str, float | str]] = []
    phase_rows: list[dict[str, float | str]] = []

    shift_names = _get_shift_names()
    shift_name = shift_names[0]
    if args_cli.shift_set and len(args_cli.shift_set) > 1:
        print(
            "[WARN] IsaacLab 在同一进程内连续切换多组参数偏移环境不稳定；"
            f"本次仅执行第一个 shift_set: {shift_name}"
        )
    print(f"[INFO] Evaluating shift set: {shift_name}")
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env_cfg = _build_shifted_env_cfg(base_env_cfg, shift_name)
    if env_cfg.seed is None:
        env_cfg.seed = agent_cfg.seed

    print(f"[INFO] Creating environment for shift '{shift_name}' (this may take a while)...")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Environment for shift '{shift_name}' is ready.")

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    command_buf = env.unwrapped.command_manager.get_command("base_velocity")
    step_dt = env.unwrapped.step_dt

    for trial_idx in range(args_cli.num_trials):
        print(f"[INFO]   Trial {trial_idx + 1}/{args_cli.num_trials}")
        obs, _ = env.reset()

        for _ in range(args_cli.settle_steps):
            command_buf[:, :3] = torch.tensor(
                [[0.0, args_cli.command_vy, args_cli.command_wz]],
                dtype=command_buf.dtype,
                device=command_buf.device,
            )
            with torch.no_grad():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

        vy_err_hist = []
        wz_err_hist = []
        pitch_abs_hist = []
        roll_abs_hist = []
        termination_reason = "time_window_complete"
        survival_steps = args_cli.record_steps

        for step_idx in range(args_cli.record_steps):
            command_buf[:, :3] = torch.tensor(
                [[0.0, args_cli.command_vy, args_cli.command_wz]],
                dtype=command_buf.dtype,
                device=command_buf.device,
            )
            with torch.no_grad():
                actions = policy(obs)
                obs, _, terminated, _ = env.step(actions)

            root_lin_vel_b = getattr(robot.data, "root_lin_vel_b", None)
            if root_lin_vel_b is None:
                root_lin_vel_b = getattr(robot.data, "root_com_lin_vel_b")
            root_ang_vel_b = getattr(robot.data, "root_ang_vel_b", None)
            if root_ang_vel_b is None:
                root_ang_vel_b = getattr(robot.data, "root_com_ang_vel_b")

            quat_w = robot.data.root_quat_w[0:1]
            roll_rad, pitch_rad, _ = euler_xyz_from_quat(quat_w)
            vy_actual = float(root_lin_vel_b[0, 1].item())
            wz_actual = float(root_ang_vel_b[0, 2].item())

            vy_err_hist.append(args_cli.command_vy - vy_actual)
            wz_err_hist.append(args_cli.command_wz - wz_actual)
            pitch_abs_hist.append(abs(float(torch.rad2deg(pitch_rad)[0].item())))
            roll_abs_hist.append(abs(float(torch.rad2deg(roll_rad)[0].item())))

            if args_cli.export_phase_csv and trial_idx == 0:
                phase_rows.append(
                    {
                        "model_name": args_cli.model_name,
                        "shift_set": shift_name,
                        "trial_idx": float(trial_idx),
                        "time_s": step_idx * step_dt,
                        "pitch_deg": float(torch.rad2deg(pitch_rad)[0].item()),
                        "pitch_rate_rad_s": float(root_ang_vel_b[0, 1].item()),
                        "roll_deg": float(torch.rad2deg(roll_rad)[0].item()),
                        "roll_rate_rad_s": float(root_ang_vel_b[0, 0].item()),
                    }
                )

            if terminated[0].item():
                survival_steps = step_idx + 1
                termination_reason = _termination_reason(env)
                break

        vy_err = torch.tensor(vy_err_hist, dtype=torch.float32)
        wz_err = torch.tensor(wz_err_hist, dtype=torch.float32)
        trial_rows.append(
            {
                "model_name": args_cli.model_name,
                "shift_set": shift_name,
                "trial_idx": float(trial_idx),
                "command_vy": float(args_cli.command_vy),
                "command_wz": float(args_cli.command_wz),
                "survived": 1.0 if survival_steps >= args_cli.record_steps else 0.0,
                "survival_steps": float(survival_steps),
                "survival_time_s": float(survival_steps * step_dt),
                "termination_reason": termination_reason,
                "rmse_vy": float(torch.sqrt(torch.mean(vy_err**2)).item()),
                "rmse_wz": float(torch.sqrt(torch.mean(wz_err**2)).item()),
                "mean_abs_pitch_deg": float(torch.tensor(pitch_abs_hist, dtype=torch.float32).mean().item()),
                "max_abs_pitch_deg": float(torch.tensor(pitch_abs_hist, dtype=torch.float32).max().item()),
                "mean_abs_roll_deg": float(torch.tensor(roll_abs_hist, dtype=torch.float32).mean().item()),
                "max_abs_roll_deg": float(torch.tensor(roll_abs_hist, dtype=torch.float32).max().item()),
            }
        )

    env.close()
    print(f"[INFO] Finished shift '{shift_name}'.")

    output_csv = _resolve_output_csv(run_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    write_header = not (args_cli.append_output and file_exists)
    write_mode = "a" if args_cli.append_output else "w"
    with output_csv.open(write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(trial_rows)
    print(f"[INFO] Parameter-shift trial CSV exported to: {output_csv}")

    if args_cli.export_phase_csv and phase_rows:
        phase_csv = _resolve_phase_csv(run_dir)
        phase_csv.parent.mkdir(parents=True, exist_ok=True)
        with phase_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()))
            writer.writeheader()
            writer.writerows(phase_rows)
        print(f"[INFO] Phase-portrait CSV exported to: {phase_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()

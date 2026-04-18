# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint for an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="Use keyboard to override the base velocity command during playback.",
)
parser.add_argument(
    "--debug-base-frame",
    action="store_true",
    default=False,
    help="Print the first robot base-frame command, velocity, and frame axes during playback.",
)
parser.add_argument(
    "--debug-interval",
    type=int,
    default=20,
    help="Number of simulation steps between debug prints when --debug-base-frame is enabled.",
)
parser.add_argument(
    "--debug-reset-reason",
    action="store_true",
    default=False,
    help="Print active termination terms for the first environment whenever it resets.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
import weakref

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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import wheel_leg.tasks  # noqa: F401


class Se2KeyboardTeleop:
    """Simple SE(2) keyboard controller for velocity commands."""

    def __init__(self, vy_scale: float = 1.4, wz_scale: float = 1.4):
        import carb
        import omni.appwindow

        self._carb = carb
        self._vy_scale = vy_scale
        self._wz_scale = wz_scale
        self._base_command = [0.0, 0.0, 0.0]
        self._reset_requested = False

        self._app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._app_window.get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        self._key_map = {
            "W": (0.0, -self._vy_scale, 0.0),
            "UP": (0.0, -self._vy_scale, 0.0),
            "S": (0.0, self._vy_scale, 0.0),
            "DOWN": (0.0, self._vy_scale, 0.0),
            "A": (0.0, 0.0, -self._wz_scale),
            "LEFT": (0.0, 0.0, -self._wz_scale),
            "D": (0.0, 0.0, self._wz_scale),
            "RIGHT": (0.0, 0.0, self._wz_scale),
        }

    def __del__(self):
        if getattr(self, "_subscription", None) is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    def __str__(self) -> str:
        return (
            "Keyboard teleop enabled:\n"
            "  W / Up: forward\n"
            "  S / Down: backward\n"
            "  A / Left: turn left\n"
            "  D / Right: turn right\n"
            "  Space: stop\n"
            "  R: reset environment"
        )

    def reset(self):
        self._base_command = [0.0, 0.0, 0.0]

    def advance(self) -> list[float]:
        return self._base_command

    def pop_reset_request(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def _on_keyboard_event(self, event, *args):
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "SPACE":
                self.reset()
            elif event.input.name == "R":
                self._reset_requested = True
            elif event.input.name in self._key_map:
                delta = self._key_map[event.input.name]
                self._base_command = [curr + inc for curr, inc in zip(self._base_command, delta)]
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._key_map:
                delta = self._key_map[event.input.name]
                self._base_command = [curr - inc for curr, inc in zip(self._base_command, delta)]
        return True


def _get_base_frame_debug(robot):
    """Collect base-frame debug values from the robot data buffers."""
    quat_w = getattr(robot.data, "root_quat_w", None)
    lin_vel_b = getattr(robot.data, "root_lin_vel_b", None)
    ang_vel_b = getattr(robot.data, "root_ang_vel_b", None)
    if quat_w is None:
        quat_w = getattr(robot.data, "body_quat_w", None)
        if quat_w is not None:
            quat_w = quat_w[:, 0, :]
    if lin_vel_b is None:
        lin_vel_b = getattr(robot.data, "root_com_lin_vel_b", None)
    if ang_vel_b is None:
        ang_vel_b = getattr(robot.data, "root_com_ang_vel_b", None)
    return quat_w, lin_vel_b, ang_vel_b


def _format_tensor3(tensor) -> str:
    values = tensor.detach().cpu().tolist()
    return f"[{values[0]: .3f}, {values[1]: .3f}, {values[2]: .3f}]"


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with an RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.keyboard and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")

    teleop = None
    teleop_command = None
    robot = env.unwrapped.scene["robot"] if args_cli.debug_base_frame else None
    if args_cli.keyboard:
        teleop = Se2KeyboardTeleop(vy_scale=1.4, wz_scale=1.4)
        teleop_command = env.unwrapped.command_manager.get_command("base_velocity")
        print(teleop)
        if env.unwrapped.num_envs != 1:
            print(
                f"[INFO] Keyboard teleop is intended for 1 environment. "
                f"Current num_envs={env.unwrapped.num_envs}; the same command will be broadcast to all envs."
            )

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        if teleop is not None:
            if teleop.pop_reset_request():
                obs, _ = env.reset()
                teleop.reset()
                continue
            command = torch.tensor(teleop.advance(), dtype=teleop_command.dtype, device=teleop_command.device)
            teleop_command[:, :3] = command.unsqueeze(0).repeat(teleop_command.shape[0], 1)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, terminated, _ = env.step(actions)

        if args_cli.debug_base_frame and timestep % max(args_cli.debug_interval, 1) == 0:
            quat_w, lin_vel_b, ang_vel_b = _get_base_frame_debug(robot)
            if quat_w is not None and lin_vel_b is not None and ang_vel_b is not None:
                base_x_w = quat_apply(quat_w[0:1], torch.tensor([[1.0, 0.0, 0.0]], device=quat_w.device))[0]
                base_y_w = quat_apply(quat_w[0:1], torch.tensor([[0.0, 1.0, 0.0]], device=quat_w.device))[0]
                current_command = None
                if teleop_command is not None:
                    current_command = teleop_command[0, :3]
                else:
                    current_command = env.unwrapped.command_manager.get_command("base_velocity")[0, :3]
                print(
                    "[BASE-DEBUG] "
                    f"cmd_b={_format_tensor3(current_command)} "
                    f"lin_vel_b={_format_tensor3(lin_vel_b[0])} "
                    f"ang_vel_b={_format_tensor3(ang_vel_b[0])} "
                    f"base_x_w={_format_tensor3(base_x_w)} "
                    f"base_y_w={_format_tensor3(base_y_w)}"
                )

        if args_cli.debug_reset_reason:
            terminated_env_ids = torch.nonzero(terminated, as_tuple=False).squeeze(-1)
            if terminated_env_ids.numel() > 0:
                first_env_id = int(terminated_env_ids[0].item())
                active_terms = env.unwrapped.termination_manager.get_active_iterable_terms(first_env_id)
                print(f"[RESET-DEBUG] env={first_env_id} active_terms={list(active_terms)}")

        timestep += 1

        if args_cli.video and timestep == args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

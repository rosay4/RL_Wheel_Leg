# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Wheel-Leg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Template-Wheel-Leg-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)


gym.register(
    id="Template-Wheel-Leg-Slope-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegSlopeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Template-Wheel-Leg-Mixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegMixedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Template-Wheel-Leg-Unstructured-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegUnstructuredEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# ==========================================
# 论文评估专用测试环境注册 (Evaluation Envs)
# ==========================================

# 1. 注册台阶评估环境 (Stairs)
gym.register(
    id="Template-Wheel-Leg-Eval-Stairs-v0",  # 测试脚本里 --task 参数填这个名字
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 指向你在 env_cfg.py 中新写的台阶评估配置类
        "env_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegEvalStairsEnvCfg",
        # 保持与训练时使用相同的 PPO runner 配置文件
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg", 
    },
)

# 2. 注册崎岖/非结构路面评估环境 (Rough)
gym.register(
    id="Template-Wheel-Leg-Eval-Rough-v0",  # 测试脚本里 --task 参数填这个名字
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 指向你在 env_cfg.py 中新写的崎岖评估配置类
        "env_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegEvalRoughEnvCfg",
        # 同上，保持 agent 配置不变
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
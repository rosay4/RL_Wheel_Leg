# Configuration index

The runtime configuration is defined by the Isaac Lab Python config classes in
`source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/`.

This directory is the human-readable index for those configurations. It does
not duplicate the runtime classes. Keep the task names in `task_catalog.yaml`
in sync with the Gym registrations when adding or removing environments.

## Runtime configuration sources

- Environment and reward configuration:
  `source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/wheel_leg_env_cfg.py`
- RSL-RL PPO configuration:
  `source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/agents/rsl_rl_ppo_cfg.py`
- Gym task registration:
  `source/wheel_leg/wheel_leg/tasks/manager_based/wheel_leg/__init__.py`

## Usage

Use the task ID from `task_catalog.yaml` with the existing scripts:

```bash
python scripts/list_envs.py
python scripts/rsl_rl/train.py --task=<TASK_ID>
python scripts/rsl_rl/play.py --task=<TASK_ID> --checkpoint=<CHECKPOINT>
```

The catalog is documentation only. The Isaac Lab registry and Python config
classes remain the source of truth for execution.

## 中文说明

运行时配置仍由 `source/wheel_leg/` 中的 Python 配置类提供，本目录不复制第二套运行时配置。`task_catalog.yaml` 只是任务索引，记录任务 ID、环境配置类、PPO 配置类和实验用途。

新增任务时，应同时更新 Gym 注册、Python 配置类和任务索引，并使用 `python scripts/list_envs.py` 检查任务是否能够被发现。

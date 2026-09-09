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

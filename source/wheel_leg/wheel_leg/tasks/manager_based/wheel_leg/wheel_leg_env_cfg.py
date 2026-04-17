# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
)
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshRandomGridTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg


ROBOT_CFG = SceneEntityCfg("robot")
WHEEL_CFG = SceneEntityCfg("robot", joint_names=[".*ankle.*"])
HIP_CFG = SceneEntityCfg("robot", joint_names=[".*hip.*"])


def get_wheel_leg_robot_cfg() -> ArticulationCfg:
    hip_actuator = ImplicitActuatorCfg(
        joint_names_expr=[".*hip.*"],
        stiffness=20.0,
        damping=0.5,
        effort_limit=5.0,
        velocity_limit=10.0,
    )
    knee_actuator = ImplicitActuatorCfg(
        joint_names_expr=[".*knee.*"],
        stiffness=0.0,
        damping=1.0,
        effort_limit=0.0,
        velocity_limit=20.0,
    )
    wheel_actuator = ImplicitActuatorCfg(
        joint_names_expr=[".*ankle.*"],
        stiffness=0.0,
        damping=0.5,
        effort_limit=10.0,
        velocity_limit=30.0,
    )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/gm_4_rosay/wheel_leg_correct/urdf/model/model.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.13),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={"hips": hip_actuator, "knees": knee_actuator, "wheels": wheel_actuator},
    )


@configclass
class WheelLegSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = get_wheel_leg_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class WheelLegRoughSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            curriculum=False,
            size=(6.0, 6.0),
            border_width=8.0,
            num_rows=6,
            num_cols=6,
            color_scheme="height",
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.2, 0.7),
            sub_terrains={
                "rough_grid": MeshRandomGridTerrainCfg(
                    proportion=1.0,
                    grid_width=0.35,
                    grid_height_range=(0.0, 0.08),
                    platform_width=1.2,
                ),
            },
        ),
    )
    robot: ArticulationCfg = get_wheel_leg_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class WheelLegSlopeSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            curriculum=False,
            size=(6.0, 6.0),
            border_width=8.0,
            num_rows=6,
            num_cols=6,
            color_scheme="height",
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.2, 0.7),
            sub_terrains={
                "slope_up": HfPyramidSlopedTerrainCfg(
                    proportion=0.5,
                    slope_range=(0.08, 0.22),
                    platform_width=2.0,
                ),
                "slope_down": HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.5,
                    slope_range=(0.05, 0.18),
                    platform_width=2.0,
                ),
            },
        ),
    )
    robot: ArticulationCfg = get_wheel_leg_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(-1.0, 1.2),
            ang_vel_z=(-1.2, 1.2),
        ),
    )


@configclass
class ActionsCfg:
    leg_joints = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*hip.*"],
        scale=0.5,
        use_default_offset=True,
    )
    wheel_joints = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*ankle.*"],
        scale=4.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-1.5, 1.5),
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.1,
            clip=(-10.0, 10.0),
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.02),
        )
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": WHEEL_CFG},
            scale=0.1,
            clip=(-20.0, 20.0),
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.15),
        )
        last_action_leg = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "leg_joints"},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
        )
        last_action_wheel = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "wheel_joints"},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.02),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
            "mass_distribution_params": (-0.25, 0.25),
            "operation": "add",
        },
    )
    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.3, 0.3),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.1),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 4.5),
        params={
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.4, 0.4),
            }
        },
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-20.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    wheel_vel_l2_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-5,
        params={"asset_cfg": WHEEL_CFG},
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": HIP_CFG},
    )
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": HIP_CFG},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.09, "asset_cfg": SceneEntityCfg("robot")},
    )
    joint_pos_limit = DoneTerm(
        func=mdp.joint_pos_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel_limit = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        params={"max_velocity": 80.0, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class WheelLegEnvCfg(ManagerBasedRLEnvCfg):
    scene: WheelLegSceneCfg = WheelLegSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    is_finite_obs = True

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 10.0
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation


@configclass
class WheelLegRoughEnvCfg(WheelLegEnvCfg):
    scene: WheelLegRoughSceneCfg = WheelLegRoughSceneCfg(num_envs=4096, env_spacing=6.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.20)
        self.events.reset_robot.params["pose_range"]["z"] = (0.06, 0.12)
        self.viewer.eye = (4.0, 4.0, 2.2)
        self.terminations.base_height.params["minimum_height"] = 0.03


@configclass
class WheelLegSlopeEnvCfg(WheelLegEnvCfg):
    scene: WheelLegSlopeSceneCfg = WheelLegSlopeSceneCfg(num_envs=4096, env_spacing=6.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.18)
        self.events.reset_robot.params["pose_range"]["z"] = (0.04, 0.10)
        self.viewer.eye = (4.0, 4.0, 2.0)
        self.terminations.base_height.params["minimum_height"] = 0.04

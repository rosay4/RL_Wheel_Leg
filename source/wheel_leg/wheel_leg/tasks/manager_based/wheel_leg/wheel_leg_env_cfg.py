# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab.envs.mdp as mdp

# =========================================================================
# 🚨 新增：全局实体配置 (SceneEntityCfg)
# 作用：将机器人拆分为具体的关节组，方便后续在观测和奖励中实现“局部读取/局部惩罚”
# =========================================================================
# 机器人的整体配置
ROBOT_CFG = SceneEntityCfg("robot")
# 仅提取车轮关节 (根据正则 ".*ankle.*")
WHEEL_CFG = SceneEntityCfg("robot", joint_names=[".*ankle.*"]) # 车轮
# 仅提取腿部关节 (根据你的正则 ".*hip.*")
HIP_CFG = SceneEntityCfg("robot", joint_names=[".*hip.*"])     # 驱动大腿

##
# 1. 我们的双轮腿机器人底层配置
##
def get_wheel_leg_robot_cfg() -> ArticulationCfg:
    # 1. 大腿(Hip)：驱动关节，强 PD 控制
    hip_actuator = ImplicitActuatorCfg(
        joint_names_expr=[".*hip.*"],  # 只控 hip
        stiffness=20.0, damping=0.5, effort_limit=5.0, velocity_limit=10.0,
    )
    # 2. 膝盖(Knee)：被动关节，0刚度，只给极小的阻尼防止乱晃
    knee_actuator = ImplicitActuatorCfg(
        joint_names_expr=[".*knee.*"], # 剥离出来
        stiffness=0.0, damping=1.0, effort_limit=0.0, velocity_limit=20.0,
    )
    # 3. 车轮(Wheel)：驱动关节，力矩控制
    wheel_actuator = ImplicitActuatorCfg(
        joint_names_expr=[".*ankle.*"], 
        stiffness=0.0, damping=0.5, effort_limit=10.0, velocity_limit=30.0,
    )
    
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/gm_4_rosay/wheel_leg_correct/urdf/model/model.usd",  # 记得改回你的路径
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.13), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0}, joint_vel={".*": 0.0},
        ),
        actuators={"hips": hip_actuator, "knees": knee_actuator, "wheels": wheel_actuator},
    )

##
# 2. 场景定义
##
@configclass
class WheelLegSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    # 把机器人加入场景，并加上正则前缀以便并行生成4096个
    robot: ArticulationCfg = get_wheel_leg_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

##
# 3. 动作、观察、奖励 (MDP)
##
@configclass
class ActionsCfg:
    # 腿部：位置控制 (输出目标角度)
    leg_joints = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*hip.*"], 
        scale=0.5,
        use_default_offset=True # 🚨 Sim-to-Real推荐：输出目标姿态基于默认站立位姿的偏移
    )
    # 车轮：力矩控制 (输出目标力矩)
    wheel_joints = mdp.JointEffortActionCfg(
        asset_name="robot", 
        joint_names=[".*ankle.*"], 
        scale=5.0 # 根据你的真实硬件力矩上限，适当放大 scale
    )

# ---------------------------------------------------------
# 彻底净化观测空间 (Sim-to-Real 的核心)
# ---------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # --- 1. IMU 观测 (物理绝对能拿到的数据) ---
        projected_gravity = ObsTerm(func=mdp.projected_gravity) # 机身俯仰/横滚状态
        base_ang_vel = ObsTerm(           # 陀螺仪角速度
            func=mdp.base_ang_vel, 
            scale=0.1, 
            clip=(-10.0, 10.0) # 防止物理异常导致角速度爆炸传入网络
        )

        # --- 2. 车轮编码器 (串口反馈能拿到的数据) ---
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": WHEEL_CFG},  # 🚨 绝对不能混入腿部速度！只读车轮
            scale=0.1, 
            clip=(-20.0, 20.0) # 🚨 防止轮子悬空时转速积分到无穷大
        )

        # --- 3. 历史动作反馈 (彻底替代腿部编码器状态) ---
        last_action_leg = ObsTerm(
            func=mdp.last_action, 
            params={"action_name": "leg_joints"}
        )
        last_action_wheel = ObsTerm(
            func=mdp.last_action, 
            params={"action_name": "wheel_joints"}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            # 🚨 杀手锏：开启历史缓冲区
            # 这意味着 MLP 的输入会自动堆叠前 5 帧的 IMU、轮速和命令
            # 极大增强对真实硬件“执行延迟”和“大腿暗箱状态”的推断能力
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale, 
        mode="reset", 
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-20.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 🚨 重要修改：防止惩罚车轮转动
    # 因为平衡车站立必须靠轮子来回转，如果不限制 asset_cfg，轮子的正常转速也会被扣分
    # 🚨 给车轮加一点点微弱的转速惩罚，目的不是阻止它平衡，而是阻止网络发现“把轮子转速加到无穷大可以卡出Bug”
    wheel_vel_l2_penalty = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-1e-5, 
        params={"asset_cfg": WHEEL_CFG}
    )

    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-0.0001,
        params={"asset_cfg": HIP_CFG}  # 只惩罚大腿的高速抽搐
    )
    
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.5, 
        params={"asset_cfg": HIP_CFG}  # 只惩罚大腿偏离默认姿态，车轮没有默认姿态
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.09, "asset_cfg": SceneEntityCfg("robot")}
    )

##
# 4. 最终环境注册
##
@configclass
class WheelLegEnvCfg(ManagerBasedRLEnvCfg):
    scene: WheelLegSceneCfg = WheelLegSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    is_finite_obs = True 

    def __post_init__(self) -> None:
        self.decimation = 4  # RL 控制频率: 120Hz / 4 = 30Hz 
        self.episode_length_s = 10.0
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
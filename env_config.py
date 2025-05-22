# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalize high joint accelerations to encourage smoother movements
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.01,  # Adjust this weight based on observed behavior
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.1, 0.4), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()



@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):

    rewards: RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        # goal_marker = CUBOID_MARKER_CFG.copy()
        # goal_marker.markers["cuboid"].scale = (0.1, 0.1, 0.1)
        # goal_marker.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False












# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import torch
# import math

# from isaaclab.assets import RigidObjectCfg # Keep this as it's used directly
# from isaaclab.sensors import FrameTransformerCfg
# from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
# from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
# from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
# from isaaclab.utils import configclass
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab.utils.math import quat_box_minus, quat_error_magnitude # For the new reward function

# from isaaclab_tasks.manager_based.manipulation.lift import mdp
# from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
# # dataclasses.MISSING is not used directly in a way that resolves the error, placeholder string is better for body_name
# # from dataclasses import MISSING # Not needed if we use a placeholder string or None with Optional type

# # Unused imports removed:
# # import isaaclab.sim as sim_utils
# # from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
# # from isaaclab.envs import ManagerBasedRLEnvCfg
# # from isaaclab.managers import CurriculumTermCfg as CurrTerm # Not used
# # from isaaclab.managers import EventTermCfg as EventTerm # Not used
# # from isaaclab.managers import TerminationTermCfg as DoneTerm # Not used
# # from isaaclab.scene import InteractiveSceneCfg # Not used
# # from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg # Not used
# # from isaaclab.markers.config import CUBOID_MARKER_CFG # Not used
# # from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG # Not used


# from isaaclab.managers import ObservationGroupCfg as ObsGroup
# from isaaclab.managers import ObservationTermCfg as ObsTerm
# from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg


# ##
# # Pre-defined configs
# ##
# from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


# # New MDP function for object placement reward
# # This would ideally be in the mdp.py file of the task
# def object_placement_reward_on_target(
#     env,
#     command_name: str,
#     robot_asset_cfg: SceneEntityCfg,
#     object_asset_cfg: SceneEntityCfg,
#     gripper_joint_names: tuple[str, str],  # e.g., ("panda_finger_joint1", "panda_finger_joint2")
#     gripper_fully_open_value: float,  # e.g., 0.04 for Franka fingers
#     dist_std: float,
#     orientation_std: float,  # in radians
#     velocity_thresh: float,  # object linear velocity should be low for stable placement
#     height_check_abs_tol: float = 0.02, # tolerance for checking if object is near target Z
# ) -> torch.Tensor:
#     """Rewards placing the object near the target pose with the gripper open and object stable."""
#     robot = env.scene[robot_asset_cfg.name]
#     object_asset = env.scene[object_asset_cfg.name]
    
#     # Target pose from command manager: (num_envs, 7) with quat as (x, y, z, w)
#     # Command is (pos, rot) where rot is (w, x, y, z) for UniformPoseCommand
#     # but command_manager.get_command typically returns root state format (pos, quat_xyzw)
#     raw_command = env.command_manager.get_command(command_name) # pos (:, :3), quat_xyzw (:, 3:7)
#     target_pos_w = raw_command[:, :3]
#     target_quat_w = raw_command[:, 3:7] # xyzw

#     object_pos_w = object_asset.data.root_state_w[:, 0:3]
#     object_quat_w = object_asset.data.root_state_w[:, 3:7]  # xyzw
#     object_vel_w = object_asset.data.root_state_w[:, 7:10]

#     # 1. Check if gripper is open
#     # This requires knowing the specific joint names and their 'open' configuration.
#     gripper_indices = []
#     for name in gripper_joint_names:
#         if name in robot.joint_names:
#             gripper_indices.append(robot.joint_names.index(name))
#         else:
#             # Fallback or error if a joint name is not found
#             # For simplicity, assume names are correct and present
#             # This part might need more robust index finding in a real scenario
#             raise ValueError(f"Gripper joint name {name} not found in robot asset.")

#     if not gripper_indices: # Should not happen if names are correct
#         is_gripper_open = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
#     else:
#         gripper_joints_pos = robot.data.joint_pos[:, gripper_indices]
#         # Assumes Franka: open is 0.04 for each. Sum is ~0.08.
#         # Check if sum of positions is close to sum of fully open values
#         # e.g., if each joint is >= 0.9 * gripper_fully_open_value
#         # A simpler check: if their sum is large enough, indicating they are spread.
#         # Target sum for two fingers: gripper_fully_open_value * 2
#         # Consider open if sum is > 90% of max open sum
#         is_gripper_open = torch.sum(gripper_joints_pos, dim=-1) > (gripper_fully_open_value * len(gripper_indices) * 0.85)


#     # 2. Check if object is stable (low velocity)
#     object_speed = torch.norm(object_vel_w, dim=-1)
#     is_object_stable = object_speed < velocity_thresh

#     # 3. Check distance to target position (especially XY)
#     pos_error_xy = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=-1)
#     pos_reward_xy = torch.exp(-torch.square(pos_error_xy) / (2 * dist_std**2))

#     # 4. Check if object is at the correct height (Z)
#     z_error = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
#     is_at_correct_height = z_error < height_check_abs_tol # more like a gate for Z
    
#     # 5. Check orientation to target
#     orientation_error_rad = quat_error_magnitude(object_quat_w, target_quat_w)
#     orientation_reward = torch.exp(-torch.square(orientation_error_rad) / (2 * orientation_std**2))
    
#     # Combine conditions: gripper open, object stable, close to target pose (XY and Z), correct orientation
#     # Reward is high if all conditions met.
#     # The is_at_correct_height acts as a gate for the Z position.
#     reward = pos_reward_xy * orientation_reward
#     final_reward = reward * is_gripper_open.float() * is_object_stable.float() * is_at_correct_height.float()
    
#     return final_reward


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

#     lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0) # This already handles "lifted to a height and same reward if higher"

#     object_goal_tracking = RewTerm(
#         func=mdp.object_goal_distance,
#         params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
#         weight=16.0,
#     )

#     object_goal_tracking_fine_grained = RewTerm(
#         func=mdp.object_goal_distance,
#         params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
#         weight=5.0,
#     )

#     # New reward for placing the object on target
#     placing_object_on_target = RewTerm(
#         func=object_placement_reward_on_target, # Using the new local function
#         weight=20.0, # Adjust weight as needed
#         params={
#             "command_name": "object_pose",
#             "robot_asset_cfg": SceneEntityCfg("robot"),
#             "object_asset_cfg": SceneEntityCfg("object"),
#             "gripper_joint_names": ("panda_finger_joint1", "panda_finger_joint2"),
#             "gripper_fully_open_value": 0.04, # Franka finger open state
#             "dist_std": 0.05, # std for XY position error
#             "orientation_std": math.radians(15.0), # std for orientation error (15 degrees in radians)
#             "velocity_thresh": 0.1, # m/s, for stability
#             "height_check_abs_tol": 0.02 # meters, for Z position check
#         }
#     )

#     # action penalty
#     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         weight=-1e-4,
#         params={"asset_cfg": SceneEntityCfg("robot")},
#     )

#     # Penalize high joint accelerations to encourage smoother movements
#     joint_acc = RewTerm(
#         func=mdp.joint_acc_l2,
#         weight=-0.01,  # Adjust this weight based on observed behavior
#         params={"asset_cfg": SceneEntityCfg("robot")},
#     )


# @configclass
# class CommandsCfg:
#     """Command terms for the MDP."""

#     object_pose = mdp.UniformPoseCommandCfg(
#         asset_name="robot",
#         # body_name expects str. Using a placeholder as it's set later.
#         # If UniformPoseCommandCfg.body_name could be Optional[str], None would be cleaner.
#         body_name="_BODY_NAME_PLACEHOLDER_",  # will be set by agent env cfg
#         resampling_time_range=(5.0, 5.0),
#         debug_vis=True,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.1, 0.4), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
#         ),
#     )


# @configclass
# class ObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#         object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
#         target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
#         actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()


# @configclass
# class FrankaCubeLiftEnvCfg(LiftEnvCfg):

#     rewards: RewardsCfg = RewardsCfg()
#     commands: CommandsCfg = CommandsCfg()
#     # observations: ObservationsCfg = ObservationsCfg() # Already in LiftEnvCfg and configured there.

#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # Set Franka as robot
#         # Assuming FRANKA_PANDA_CFG.replace exists and works as expected for configclass instances
#         self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#         # Set actions for the specific robot type (franka)
#         self.actions.arm_action = mdp.JointPositionActionCfg(
#             asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
#         )
#         self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
#             asset_name="robot",
#             joint_names=["panda_finger.*"],
#             open_command_expr={"panda_finger_.*": 0.04},
#             close_command_expr={"panda_finger_.*": 0.0},
#         )
#         # Set the body name for the end effector for commands that need it
#         self.commands.object_pose.body_name = "panda_hand"

#         # Set Cube as object
#         self.scene.object = RigidObjectCfg(
#             prim_path="{ENV_REGEX_NS}/Object",
#             init_state=RigidObjectCfg.InitialStateCfg(
#                 pos=(0.5, 0.0, 0.055), # Changed list to tuple
#                 rot=(1.0, 0.0, 0.0, 0.0) # Changed list to tuple, ensure floats
#             ),
#             spawn=UsdFileCfg(
#                 usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
#                 scale=(0.8, 0.8, 0.8),
#                 rigid_props=RigidBodyPropertiesCfg(
#                     solver_position_iteration_count=16,
#                     solver_velocity_iteration_count=1,
#                     max_angular_velocity=1000.0,
#                     max_linear_velocity=1000.0,
#                     max_depenetration_velocity=5.0,
#                     disable_gravity=False,
#                 ),
#             ),
#         )

#         # Listens to the required transforms
#         # Assuming FRAME_MARKER_CFG.copy() exists and works as expected
#         marker_cfg = FRAME_MARKER_CFG.copy()
#         marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
#         marker_cfg.prim_path = "/Visuals/FrameTransformer"

#         self.scene.ee_frame = FrameTransformerCfg(
#             prim_path="{ENV_REGEX_NS}/Robot/panda_link0", # Base frame for EE transform
#             debug_vis=False, # Setting to False, was True in LiftEnvCfg example
#             visualizer_cfg=marker_cfg,
#             target_frames=[
#                 FrameTransformerCfg.FrameCfg(
#                     prim_path="{ENV_REGEX_NS}/Robot/panda_hand", # Target frame
#                     name="end_effector",
#                     offset=OffsetCfg(
#                         pos=(0.0, 0.0, 0.1034), # Changed list to tuple
#                         # rot defaults to (1.0, 0.0, 0.0, 0.0) which is identity quat (w,x,y,z)
#                     ),
#                 ),
#             ],
#         )


# @configclass
# class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # disable randomization for play
#         if self.observations.policy is not None: # Check if policy group exists
#             self.observations.policy.enable_corruption = False

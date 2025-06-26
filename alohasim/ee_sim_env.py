import numpy as np
import collections
import os
from pyquaternion import Quaternion

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """

    # create task
    if task_name == 'sim_transfer_cube':
        task = TransferCubeEETask(random=False)
    elif task_name == "sim_pickplace_bottle":
        task = PickPlaceBottleEETask(random=False)
    else:
        raise NotImplementedError
    
    # xml name should be same as task name
    xml_path = os.path.join(XML_DIR, f'eetask_{task_name[4:]}.xml')

    # create physics and environment
    physics = mujoco.Physics.from_xml_path(xml_path)
    env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                n_sub_steps=None, flat_observation=False)
    
    return env


class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper control
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset to start joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers (print_info.py):
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['left/gripper_link']
        #     get env._physics.named.data.xquat['left/gripper_link']
        #     repeat the same for right side
        # (3) record the mocap position and quaternion of the left and right end effectors
        #     (add x axis offset to the mocap)

        # reset left mocap
        np.copyto(physics.data.mocap_pos[0], [-0.31718881 + 0.1, -0.0675, 0.31525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # reset right mocap
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881 - 0.1, -0.0675, 0.31525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)

        # cameras for aloha2
        cam_height = 480
        cam_width = 640
        obs['images'] = dict()
        obs['images']['overhead_cam'] = physics.render(height=cam_height, width=cam_width, camera_id='overhead_cam')
        obs['images']['worms_eye_cam'] = physics.render(height=cam_height, width=cam_width, camera_id='worms_eye_cam')
        obs['images']['wrist_cam_left'] = physics.render(height=cam_height, width=cam_width, camera_id='wrist_cam_left')
        obs['images']['wrist_cam_right'] = physics.render(height=cam_height, width=cam_width, camera_id='wrist_cam_right')
        # obs['images']['teleoperator_pov'] = physics.render(height=cam_height, width=cam_width, camera_id='teleoperator_pov')
        # obs['images']['collaborator_pov'] = physics.render(height=cam_height, width=cam_width, camera_id='collaborator_pov')

        # used in scripted policy
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class PickPlaceBottleEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)

        # transfer cube reward definition:
        # reward = 1: right gripper touch bottle
        # reward = 2: right girpper lift bottle
        # reward = 3: bottle touch box
        # reward = 4: bottle touch box & bottle not touch right gripper
        self.max_reward = 4

        # define relevant collision names
        self.left_gripper_collision_names = {"left/left_g", "left/left_g0", "left/left_g1", "left/left_g2",
                                        "left/right_g", "left/right_g0", "left/right_g1", "left/right_g2"}
        self.right_gripper_collision_names = {"right/left_g", "right/left_g0", "right/left_g1", "right/left_g2",
                                         "right/right_g", "right/right_g0", "right/right_g1", "right/right_g2",}
        self.box_collision_name = "box_bottom"
        self.bottle_collision_name = "pill_bottle"
        self.table_collision_name = "table"

    @staticmethod
    def init_bottle_pose():
        # initial xyz
        x, y, z = 0.3, -0.19, 0.05
        bottle_position = np.array([x, y, z])

        # initial rotation
        rotation_deg = 5
        bottle_quat = Quaternion()
        bottle_quat = bottle_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=rotation_deg)

        return np.concatenate([bottle_position, bottle_quat.elements])

    @staticmethod
    def sample_bottle_pose():
        # sample xyz
        x_range = [-0.1, 0.3]
        y_range = [-0.19, 0.07]
        z_range = [0.05, 0.05]
        ranges = np.vstack([x_range, y_range, z_range])
        bottle_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

        # sample rotation
        degree_range = [0, 180]
        bottle_rotation_degree = np.random.uniform(degree_range[0], degree_range[1])
        bottle_quat = Quaternion()
        bottle_quat = bottle_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=bottle_rotation_degree)

        print(f"sampled bottle position: {bottle_position}, rotation: {bottle_rotation_degree} degree.")

        return np.concatenate([bottle_position, bottle_quat.elements])

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)

        # box is fixed, only randomly sample bottle pose
        bottle_pose = self.sample_bottle_pose()
        bottle_start_idx = physics.model.name2id('bottle_joint', 'joint')
        np.copyto(physics.data.qpos[bottle_start_idx : bottle_start_idx + 7], bottle_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # bottle joint [0:7]
        # box_joint [7:14]
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # get all contact pairs as list of tuples
        all_contact_pairs = []
        box_contact_set = set()
        bottle_contact_set = set()
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

            if self.box_collision_name == contact_pair[0]:
                box_contact_set.add(contact_pair[1])
            if self.box_collision_name == contact_pair[1]:
                box_contact_set.add(contact_pair[0])

            if self.bottle_collision_name == contact_pair[0]:
                bottle_contact_set.add(contact_pair[1])
            if self.bottle_collision_name == contact_pair[1]:
                bottle_contact_set.add(contact_pair[0])

        # touch determination
        touch_right_gripper = len(bottle_contact_set.intersection(self.right_gripper_collision_names)) > 0
        touch_box = self.box_collision_name in bottle_contact_set
        touch_table = self.table_collision_name in bottle_contact_set

        # calculate rewated based on contact state
        reward = 0
        if touch_right_gripper: # bottle touched
            reward = 1
        if touch_right_gripper and not touch_table: # box lifted
            reward = 2
        if touch_box: # attempted place
            reward = 3
        if touch_box and not touch_right_gripper: # successful place
            reward = 4
        return reward


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)

        # transfer cube reward definition:
        # reward = 1: right gripper touch box
        # reward = 2: right girpper lift box
        # reward = 3: left gripper touch box
        # reward = 4: left gripper touch box & box not touch table
        self.max_reward = 4

        # define relevant collision names
        self.left_gripper_collision_names = {"left/left_g", "left/left_g0", "left/left_g1", "left/left_g2",
                                        "left/right_g", "left/right_g0", "left/right_g1", "left/right_g2"}
        self.right_gripper_collision_names = {"right/left_g", "right/left_g0", "right/left_g1", "right/left_g2",
                                         "right/right_g", "right/right_g0", "right/right_g1", "right/right_g2",}
        self.box_collision_name = "red_box"
        self.table_collision_name = "table"

    @staticmethod
    def sample_box_pose():
        x_range = [0.0, 0.2]
        y_range = [-0.1, 0.1]
        z_range = [0.05, 0.05]

        ranges = np.vstack([x_range, y_range, z_range])
        cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

        cube_quat = np.array([1, 0, 0, 0])
        return np.concatenate([cube_position, cube_quat])

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        print(f"randomized cube position to {cube_pose[:3]}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # get all contact pairs as list of tuples
        all_contact_pairs = []
        box_contact_set = set() # set of geoms contacted with red_box
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

            if self.box_collision_name == contact_pair[0]:
                box_contact_set.add(contact_pair[1])
            if self.box_collision_name == contact_pair[1]:
                box_contact_set.add(contact_pair[0])

        # touch determination
        touch_left_gripper = len(box_contact_set.intersection(self.left_gripper_collision_names)) > 0
        touch_right_gripper = len(box_contact_set.intersection(self.right_gripper_collision_names)) > 0
        touch_table = self.table_collision_name in box_contact_set

        # calculate rewated based on contact state
        reward = 0
        if touch_right_gripper: # box touched
            reward = 1
        if touch_right_gripper and not touch_table: # box lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward

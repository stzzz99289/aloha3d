import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from pyquaternion import Quaternion

from constants import DT, XML_DIR, START_ARM_POSE, START_CTRL
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
# from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

ENV_STATE = [None] # to be changed from outside


def sample_env_state(task_name, **kwargs):
    if task_name == 'sim_transfer_cube':
        # env_state dim: 7 (cube xyz + quat)
        raise NotImplementedError
    elif task_name == "sim_pickplace_bottle":
        # env_state dim: 14 (bottle xyz + quat, box xyz + quat)
        # sample bottle xyz
        x_range = kwargs['x_range'] if 'x_range' in kwargs else [-0.1, 0.3]
        y_range = kwargs['y_range'] if 'y_range' in kwargs else [-0.19, 0.07]
        z_range = [0.05, 0.05]
        ranges = np.vstack([x_range, y_range, z_range])
        bottle_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

        # sample bottle rotation
        degree_range = [0, 180]
        bottle_rotation_degree = np.random.uniform(degree_range[0], degree_range[1])
        bottle_quat = Quaternion()
        bottle_quat = bottle_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=bottle_rotation_degree)

        # fixed box pose
        box_pose = [0, 0.215, 0, 1, 0, 0, 0]

        # sampled env state
        env_state = np.concatenate([bottle_position, bottle_quat.elements, box_pose])
        return env_state
    else:
        raise NotImplementedError


def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
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
        task = TransferCubeTask(random=False)
    elif task_name == "sim_pickplace_bottle":
        task = PickPlaceBottleTask(random=False)
    else:
        raise NotImplementedError
    
    # xml name should be same as task name
    xml_path = os.path.join(XML_DIR, f'task_{task_name[4:]}.xml')

    # create physics and environment
    physics = mujoco.Physics.from_xml_path(xml_path)
    env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                n_sub_steps=None, flat_observation=False)
    
    return env


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action]
        full_right_gripper_action = [right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

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

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PickPlaceBottleTask(BimanualViperXTask):
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

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set ENV_STATE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_CTRL)
            assert ENV_STATE[0] is not None
            physics.named.data.qpos[16:] = ENV_STATE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
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


class TransferCubeTask(BimanualViperXTask):
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

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_CTRL)
            assert ENV_STATE[0] is not None
            physics.named.data.qpos[16:] = ENV_STATE[0]
            # print(f"{BOX_POSE=}")
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

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert ENV_STATE[0] is not None
            physics.named.data.qpos[16:] = ENV_STATE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward





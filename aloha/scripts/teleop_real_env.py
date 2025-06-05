import argparse
from aloha.real_env import get_action, make_real_env
from aloha.robot_utils import (
    disable_gravity_compensation,
    ImageRecorder,
    load_yaml_file,
    move_grippers,
    torque_on,
    LEFT_FOLLOWER_GRIPPER_JOINT_OPEN,
    RIGHT_FOLLOWER_GRIPPER_JOINT_OPEN,
)
import cv2
import h5py
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
import numpy as np
import os
from pathlib import Path
import time
from typing import Dict, List
from tqdm import tqdm
from utils import opening_ceremony, press_to_start
import matplotlib.pyplot as plt

def main(args: Dict[str, any]) -> None:
    # Retrieve arguments and configuration settings
    torque_base = False
    gravity_compensation = False
    robot_base = args.get("robot", "")
    base_path = Path(__file__).resolve().parent.parent / "config"

    # Load robot and task configurations from YAML files
    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})

    # Determine if the robot has a mobile base and set the control frequency
    IS_MOBILE = config.get("base", False)
    DT = 1 / config.get("fps", 50)

    # Initialize the ROS node and robot environment
    onscreen_render = True
    render_cam = 'camera_wrist_left'
    node = create_interbotix_global_node("aloha")
    env = make_real_env(
        node=node,
        setup_robots=False,
        setup_base=IS_MOBILE,
        torque_base=torque_base,
        config=config,
    )
    robot_startup(node)
    opening_ceremony(env.robots, dt=DT)
    press_to_start(env.robots, dt=DT, gravity_compensation=gravity_compensation)

    time.sleep(1)
    ts = env.reset(fake=True)
    episode = [ts]

    # visualization setup
    if onscreen_render:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plt_color = ax1.imshow(ts.observation['images'][f"{render_cam}_color_image"])
        ax1.set_title('Color Image')

        depth_image = ts.observation['images'][f"{render_cam}_aligned_depth_image"]
        normalized_depth = np.clip(depth_image / 500.0, 0, 1)  # normalize to [0,1] range
        plt_depth = ax2.imshow(normalized_depth, cmap='viridis')
        ax2.set_title('Depth Image')
        # plt.colorbar(plt_depth, ax=ax2)
        plt.ion()

    while True:
        action = get_action(env.robots)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_color.set_data(ts.observation['images'][f"{render_cam}_color_image"])
            depth_image = ts.observation['images'][f"{render_cam}_aligned_depth_image"]
            normalized_depth = np.clip(depth_image / 0.5, 0, 1)
            plt_depth.set_data(normalized_depth)
            plt.pause(env.dt)
        else:
            time.sleep(env.dt)

if __name__ == "__main__":
    # Argument parser to manage command-line inputs
    parser = argparse.ArgumentParser(
        description="Launches robot teleoperation with specified parameters.")

    # Robot setup configuration: required
    parser.add_argument(
        "-r",
        "--robot",
        action="store",
        type=str,
        help="Robot setup configuration (e.g., aloha_solo, aloha_static, aloha_mobile).",
        required=True,
    )

    # Execute the main function with parsed arguments
    main(vars(parser.parse_args()))
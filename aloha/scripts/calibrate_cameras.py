import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from aloha.robot_utils import (
    move_arms,
    move_grippers,
    load_yaml_file,
    LEFT_FOLLOWER_GRIPPER_JOINT_OPEN,
    RIGHT_FOLLOWER_GRIPPER_JOINT_OPEN,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_startup,
    robot_shutdown,
)
from utils import opening_ceremony, press_to_start
from typing import Dict
import argparse
from aloha.real_env import make_real_env
import numpy as np
import os
import time
from typing import Dict
from tqdm import tqdm
from utils import opening_ceremony, press_to_start
from pathlib import Path
import cv2
import rclpy

def main(args: Dict[str, any]) -> None:
    """
    Main function for executing a robot teleoperation task based on configuration parameters.
    Handles dataset setup, task configuration, and runs the teleoperation loop to capture episodes.

    :param args: Dictionary of arguments, expected keys:
        - "enable_base_torque" (bool): Whether to enable torque on the base (for mobile robots).
        - "gravity_compensation" (bool): Whether to enable gravity compensation for leader robots.
        - "robot" (str): Robot setup configuration (e.g., 'aloha_solo', 'aloha_static', 'aloha_mobile').
        - "task_name" (str): Task name used to fetch specific task configuration.
        - "episode_idx" (Optional[int]): Episode index for dataset naming; if None, auto-indexing is used.
    """
    robot_base = args.get("robot", "")
    base_path = Path(__file__).resolve().parent.parent / "config"
    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})

    # Determine if the robot has a mobile base and set the control frequency
    IS_MOBILE = False
    DT = 1 / 50

    # Initialize the ROS node and robot environment
    node = create_interbotix_global_node("aloha")
    env = make_real_env(
        node=node,
        setup_robots=False,
        setup_base=IS_MOBILE,
        torque_base=False,
        config=config,
    )
    robot_startup(node)

    # Read calibration positions from file
    calibration_file = os.path.join(base_path, f"calibration_positions_{args['mode']}.txt")
    calibration_positions = []
    with open(calibration_file, 'r') as f:
        for line in f:
            # Remove brackets and split by comma
            clean_line = line.strip('[] \n').split(',')
            # Convert each string to float, removing quotes
            row = [float(x.strip("' ")) for x in clean_line]
            calibration_positions.append(row)
    calibration_positions = np.array(calibration_positions)
    num_positions = len(calibration_positions)

    # Move robots to starting position and wait for user to start
    opening_ceremony(env.robots, dt=DT)

    # Begin data collection
    time.sleep(2.0)
    ts = env.reset(fake=True)

    # open grippers
    move_grippers(
        bot_list=[env.robots['follower_left'], env.robots['follower_right']],
        dt=DT,
        target_pose_list=[LEFT_FOLLOWER_GRIPPER_JOINT_OPEN, RIGHT_FOLLOWER_GRIPPER_JOINT_OPEN],
        moving_time=2.0,
    )

    # data collection
    camera_names = ['camera_wrist_left', 'camera_wrist_right', 'camera_low', 'camera_high']
    for camera_name in camera_names:
        os.makedirs(f'calibration_info_{args["mode"]}/{camera_name}/images', exist_ok=True)
    os.makedirs(f'calibration_info_{args["mode"]}/left_arm/joint_positions', exist_ok=True)
    os.makedirs(f'calibration_info_{args["mode"]}/right_arm/joint_positions', exist_ok=True)
    for i in tqdm(range(num_positions)):
        position = calibration_positions[i]
        time.sleep(2.0)
        move_arms(
            bot_list=[env.robots['follower_left'], env.robots['follower_right']],
            dt=DT,
            target_pose_list=[position, position],
            moving_time=2.0,
        )
        time.sleep(2.0)
        obs = env.get_observation()
        for camera_name in camera_names:
            captured_image = obs['images'][camera_name + '_color_image']
            cv2.imwrite(f'calibration_info_{args["mode"]}/{camera_name}/images/{i:02d}.png', cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
        np.save(f'calibration_info_{args["mode"]}/left_arm/joint_positions/{i:02d}.npy', env.robots['follower_left'].arm.get_joint_positions())
        np.save(f'calibration_info_{args["mode"]}/right_arm/joint_positions/{i:02d}.npy', env.robots['follower_right'].arm.get_joint_positions())
    robot_shutdown()

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

    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        type=str,
        help="calibrate with the chessboard lay or stand",
    )

    # Execute the main function with parsed arguments
    main(vars(parser.parse_args()))
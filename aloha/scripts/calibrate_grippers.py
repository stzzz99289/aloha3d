#!/usr/bin/env python3
import argparse
from aloha.robot_utils import (
    load_yaml_file,
    LEFT_LEADER2FOLLOWER_JOINT_FN,
    RIGHT_LEADER2FOLLOWER_JOINT_FN,
    LEFT_LEADER_GRIPPER_JOINT_NORMALIZE_FN,
    RIGHT_LEADER_GRIPPER_JOINT_NORMALIZE_FN
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from pathlib import Path
import rclpy
from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
from typing import Dict
from utils import opening_ceremony

def grippers_torque_off(robots: Dict[str, InterbotixManipulatorXS],
                        ) -> None:
    """
    Torque off all grippers for calibration.
    """
    for name, bot in robots.items():
        bot.core.robot_torque_enable('single', 'gripper', False)

def main(args: dict) -> None:
    """
    Main teleoperation setup function.

    :param args: Dictionary containing parsed arguments including gravity compensation
                 and robot configuration.
    """
    node = create_interbotix_global_node('aloha')

    # Load robot configuration
    robot_base = args.get('robot', '')

    # Base path of the config directory using absolute path
    base_path = Path(__file__).resolve().parent.parent / "config"

    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})
    dt = 1 / config.get('fps', 30)

    # Initialize dictionary for robot instances
    robots = {}

    # Create leader arms from configuration
    for leader in config.get('leader_arms', []):
        robot_instance = InterbotixManipulatorXS(
            robot_model=leader['model'],
            robot_name=leader['name'],
            node=node,
            iterative_update_fk=False,
        )
        robots[leader['name']] = robot_instance

    # Create follower arms from configuration
    for follower in config.get('follower_arms', []):
        robot_instance = InterbotixManipulatorXS(
            robot_model=follower['model'],
            robot_name=follower['name'],
            node=node,
            iterative_update_fk=False,
        )
        robots[follower['name']] = robot_instance

    # Startup and initialize robot sequence
    robot_startup(node)
    opening_ceremony(robots, dt)

    # torque off leader and follower grippers for calibration
    grippers_torque_off(robots)

    # Main teleoperation loop
    lines = ["", ""]
    for line in lines:
        print(line)

    while rclpy.ok():
        for leader_name, leader_bot in robots.items():
            if 'leader' in leader_name:
                suffix = leader_name.replace('leader', '')
                follower_name = f'follower{suffix}'
                follower_bot = robots.get(follower_name)

                if follower_bot:
                    # Sync arm joint positions and gripper positions
                    leader_state_joints = leader_bot.arm.get_joint_positions()
                    follower_bot.arm.set_joint_positions(leader_state_joints, blocking=False)

                    leader_gripper_position = leader_bot.gripper.get_gripper_position()
                    if "left" in leader_name:
                        normalized_gripper_command = LEFT_LEADER_GRIPPER_JOINT_NORMALIZE_FN(leader_gripper_position)
                        follower_gripper_command = LEFT_LEADER2FOLLOWER_JOINT_FN(leader_gripper_position)
                    elif "right" in leader_name:
                        normalized_gripper_command = RIGHT_LEADER_GRIPPER_JOINT_NORMALIZE_FN(leader_gripper_position)
                        follower_gripper_command = RIGHT_LEADER2FOLLOWER_JOINT_FN(leader_gripper_position)
                    follower_gripper_position = follower_bot.gripper.get_gripper_position()

                    if "left" in leader_name:
                        lines[0] = f"[left] leader:{leader_gripper_position:7.4f}, normalized cmd: {normalized_gripper_command:7.4f}, follower cmd:{follower_gripper_command:7.4f}, follower:{follower_gripper_position:7.4f}"
                    elif "right" in leader_name:
                        lines[1] = f"[right] leader:{leader_gripper_position:7.4f}, normalized cmd: {normalized_gripper_command:7.4f}, follower cmd:{follower_gripper_command:7.4f}, follower:{follower_gripper_position:7.4f}"

        # print calibration info in console
        print(f"\033[{len(lines)}A", end='')
        for line in lines:
            print(line)
            print("\033[K", end='')  # Clear rest of the line

        # Sleep for the DT duration
        DT_DURATION = Duration(seconds=0, nanoseconds=dt * S_TO_NS)
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--robot',
        required=True,
        help='Specify the robot configuration to use: aloha_solo, aloha_static, or aloha_mobile.'
    )
    main(vars(parser.parse_args()))

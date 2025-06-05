#!/usr/bin/env python3

import argparse
from aloha.robot_utils import (
    load_yaml_file,
    LEFT_LEADER2FOLLOWER_JOINT_FN,
    RIGHT_LEADER2FOLLOWER_JOINT_FN,
    LEFT_LEADER_GRIPPER_JOINT_NORMALIZE_FN,
    RIGHT_LEADER_GRIPPER_JOINT_NORMALIZE_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
)

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from pathlib import Path
import rclpy
from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
from utils import opening_ceremony, press_to_start

def main(args: dict) -> None:
    """
    Main teleoperation setup function.

    :param args: Dictionary containing parsed arguments including gravity compensation
                 and robot configuration.
    """
    gravity_compensation = args.get('gravity_compensation', False)
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
    press_to_start(robots, dt, gravity_compensation)

    # Define gripper command objects for each follower
    gripper_commands = {
        follower_name: JointSingleCommand(name='gripper') for follower_name in robots if 'follower' in follower_name
    }

    # Main teleoperation loop
    mirror = True
    while rclpy.ok():
        for leader_name, leader_bot in robots.items():
            if 'leader' in leader_name:
                suffix = leader_name.replace('leader', '')
                follower_name = f'follower{suffix}'
                follower_bot = robots.get(follower_name)

                if follower_bot:
                    # Sync arm joint positions and gripper positions
                    if 'left' in leader_name:
                        left_leader_state_joints = leader_bot.arm.get_joint_positions()
                        if mirror:
                            follower_bot.arm.set_joint_positions(right_leader_state_joints, blocking=False)
                        else:
                            follower_bot.arm.set_joint_positions(left_leader_state_joints, blocking=False)
                    elif 'right' in leader_name:
                        right_leader_state_joints = leader_bot.arm.get_joint_positions()
                        follower_bot.arm.set_joint_positions(right_leader_state_joints, blocking=False)

                    # Sync gripper positions
                    gripper_command = gripper_commands[follower_name]
                    leader_gripper_position = leader_bot.gripper.get_gripper_position()
                    if 'left' in leader_name:
                        normalized_command = LEFT_LEADER_GRIPPER_JOINT_NORMALIZE_FN(leader_gripper_position)
                        follower_gripper_command = LEFT_LEADER2FOLLOWER_JOINT_FN(leader_gripper_position)
                    elif 'right' in leader_name:
                        normalized_command = RIGHT_LEADER_GRIPPER_JOINT_NORMALIZE_FN(leader_gripper_position)
                        follower_gripper_command = RIGHT_LEADER2FOLLOWER_JOINT_FN(leader_gripper_position)
                    gripper_command.cmd = follower_gripper_command
                    follower_bot.gripper.core.pub_single.publish(gripper_command)

                    if normalized_command < LEADER_GRIPPER_CLOSE_THRESH:
                        follower_arm_joint_positions = follower_bot.arm.get_joint_positions()
                        follower_arm_joint_positions = [f"{x:6.3f}" for x in follower_arm_joint_positions]
                        print(f"{follower_name} arm joint positions: {follower_arm_joint_positions}")

        # Sleep for the DT duration
        DT_DURATION = Duration(seconds=0, nanoseconds=dt * S_TO_NS)
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--gravity_compensation',
        action='store_true',
        help='If set, gravity compensation will be enabled for the leader robots when teleop starts.',
    )
    parser.add_argument(
        '-r', '--robot',
        required=True,
        help='Specify the robot configuration to use: aloha_solo, aloha_static, or aloha_mobile.'
    )
    main(vars(parser.parse_args()))

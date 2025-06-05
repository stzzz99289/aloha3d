from aloha.robot_utils import (
    enable_gravity_compensation,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
    LEFT_FOLLOWER_GRIPPER_JOINT_CLOSE,
    RIGHT_FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEFT_LEADER_GRIPPER_JOINT_MID,
    RIGHT_LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)

from interbotix_common_modules.common_robot.robot import (
    get_interbotix_global_node,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import rclpy
from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
from typing import Dict

def opening_ceremony(robots: Dict[str, InterbotixManipulatorXS],
                     dt: float,
                     ) -> None:
    """
    Move all leader-follower pairs of robots to a starting pose for demonstration.

    :param robots: Dictionary containing robot instances categorized as 'leader' or 'follower'
    :param dt: Time interval (in seconds) for each movement step
    """
    # Separate leader and follower robots
    leader_bots = {name: bot for name,
                   bot in robots.items() if 'leader' in name}
    follower_bots = {name: bot for name,
                     bot in robots.items() if 'follower' in name}

    # Initialize an empty list to store matched pairs of leader and follower robots
    pairs = {}

    # Create dictionaries mapping suffixes to leader and follower robots
    leader_suffixes = {name.split(
        '_', 1)[1]: bot for name, bot in leader_bots.items()}
    follower_suffixes = {name.split(
        '_', 1)[1]: bot for name, bot in follower_bots.items()}

    # Pair leader and follower robots based on matching suffixes
    for suffix, leader_bot in leader_suffixes.items():
        if suffix in follower_suffixes:
            # If matching follower exists, pair it with the leader
            follower_bot = follower_suffixes.pop(suffix)
            # pairs.append((leader_bot, follower_bot))
            pairs[suffix] = (leader_bot, follower_bot)
        else:
            # Raise an error if there’s an unmatched leader suffix
            raise ValueError(
                f"Unmatched leader suffix '{suffix}' found. Every leader should have a corresponding follower with the same suffix.")

    # Check if any unmatched followers remain after pairing
    if follower_suffixes:
        unmatched_suffixes = ', '.join(follower_suffixes.keys())
        raise ValueError(
            f"Unmatched follower suffix(es) found: {unmatched_suffixes}. Every follower should have a corresponding leader with the same suffix.")

    # Ensure at least one leader-follower pair was created
    if not pairs:
        raise ValueError(
            "No valid leader-follower pairs found in the robot dictionary.")

    # Initialize each leader-follower pair
    for suffix in pairs:
        leader_bot, follower_bot = pairs[suffix]

        # Reboot gripper motors and set operating modes
        follower_bot.core.robot_reboot_motors('single', 'gripper', True)
        follower_bot.core.robot_set_operating_modes('group', 'arm', 'position')
        follower_bot.core.robot_set_operating_modes(
            'single', 'gripper', 'current_based_position')
        leader_bot.core.robot_set_operating_modes('group', 'arm', 'position')
        leader_bot.core.robot_set_operating_modes(
            'single', 'gripper', 'position')
        follower_bot.core.robot_set_motor_registers(
            'single', 'gripper', 'current_limit', 300)

        # Enable torque for leader and follower
        torque_on(follower_bot)
        torque_on(leader_bot)

        # Move arms to starting position
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms(
            bot_list=[leader_bot, follower_bot],
            dt=dt,
            target_pose_list=[start_arm_qpos] * 2,
            moving_time=4.0,
        )

        # Move grippers to starting position
        if suffix == "right":
            move_grippers(
                [leader_bot, follower_bot],
                [RIGHT_LEADER_GRIPPER_JOINT_MID, RIGHT_FOLLOWER_GRIPPER_JOINT_CLOSE],
                moving_time=0.5,
                dt=dt,
            )
        elif suffix == "left":
            move_grippers(
                [leader_bot, follower_bot],
                [LEFT_LEADER_GRIPPER_JOINT_MID, LEFT_FOLLOWER_GRIPPER_JOINT_CLOSE],
                moving_time=0.5,
                dt=dt,
            )


def press_to_start(robots: Dict[str, InterbotixManipulatorXS],
                   dt: float,
                   gravity_compensation: bool,
                   ) -> None:
    """
    Wait for the user to close the grippers on all leader robots to start teleoperation.

    :param robots: Dictionary containing robot instances categorized as 'leader' or 'follower'
    :param dt: Time interval (in seconds) for each movement step
    :param gravity_compensation: Boolean flag to enable gravity compensation on leaders
    """
    # Extract leader bots from the robots dictionary
    leader_bots = {name: bot for name,
                   bot in robots.items() if 'leader' in name}

    # Disable torque for gripper joint of each leader bot to allow user movement
    for leader_bot in leader_bots.values():
        leader_bot.core.robot_torque_enable('single', 'gripper', False)

    print('Close the grippers to start')

    # Wait for the user to close the grippers on all leader robots
    pressed = False
    while rclpy.ok() and not pressed:
        pressed = all(
            get_arm_gripper_positions(leader_bot) < LEADER_GRIPPER_CLOSE_THRESH
            for leader_bot in leader_bots.values()
        )
        DT_DURATION = Duration(seconds=0, nanoseconds=dt * S_TO_NS)
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    # Enable gravity compensation or turn off torque based on the parameter
    for leader_bot in leader_bots.values():
        if gravity_compensation:
            enable_gravity_compensation(leader_bot)
        else:
            torque_off(leader_bot)

    print('Started!')


#!/bin/bash
source /opt/ros/humble/setup.bash # configure ROS system install environment
source ~/interbotix_ws/install/setup.bash # configure ROS workspace environment
python3 ~/interbotix_ws/src/aloha/scripts/teleop.py -r aloha_stationary_arms

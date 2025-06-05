#!/bin/bash
mode="$1"
source /opt/ros/humble/setup.bash # configure ROS system install environment
source ~/interbotix_ws/install/setup.bash # configure ROS workspace environment
python3 ~/interbotix_ws/src/aloha/scripts/calibrate_cameras.py -r aloha_stationary -m ${mode}
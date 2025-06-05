#!/bin/bash
source /opt/ros/humble/setup.bash # configure ROS system install environment
source ~/interbotix_ws/install/setup.bash # configure ROS workspace environment
python3 ~/interbotix_ws/src/aloha/scripts/teleop_real_env.py \
    --robot aloha_stationary \

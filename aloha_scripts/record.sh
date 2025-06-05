#!/bin/bash
task_name="$1"
episode_idx="$2"

source /opt/ros/humble/setup.bash # configure ROS system install environment
source ~/interbotix_ws/install/setup.bash # configure ROS workspace environment
python3 ~/interbotix_ws/src/aloha/scripts/record_episodes.py \
      --task_name ${task_name} \
      --episode_idx ${episode_idx} \
      --robot aloha_stationary
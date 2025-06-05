#!/bin/bash
task_name="s2"
episode_idx="3"

source /opt/ros/humble/setup.bash # configure ROS system install environment
source ~/interbotix_ws/install/setup.bash # configure ROS workspace environment
python3 ~/interbotix_ws/src/aloha/scripts/replay_episodes.py \
    --robot aloha_stationary \
    --dataset_dir ~/aloha_data/${task_name} \
    --episode_idx ${episode_idx}
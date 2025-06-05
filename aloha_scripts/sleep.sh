#!/bin/bash
export INTERBOTIX_ALOHA_IS_MOBILE=false # true for Mobile, false for Stationary
source /opt/ros/humble/setup.bash # configure ROS system install environment
source ~/interbotix_ws/install/setup.bash # configure ROS workspace environment
python3 ~/interbotix_ws/src/aloha/scripts/sleep.py -r aloha_stationary_arms

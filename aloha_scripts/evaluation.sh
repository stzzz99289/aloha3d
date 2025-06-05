 #!/bin/bash
task_name="catch_box"
source ~/act/bin/activate
source /opt/ros/humble/setup.bash # configure ROS system install environment
source ~/interbotix_ws/install/setup.bash # configure ROS workspace environment
python3 ~/interbotix_ws/src/act_training_evaluation/act/imitate_episodes.py \
  --task_name ${task_name} \
  --ckpt_dir ~/aloha_ckpt/${task_name} \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0 \
  --eval \
  --temporal_agg
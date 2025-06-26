import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import csv
from time import sleep
from pyquaternion import Quaternion

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, ENV_STATE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy, PickPlaceBottlePolicy

def main(args):
    """
    Generate demonstration data in simulation.
    1. Rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    2. Replace the gripper joint positions with the commanded joint position.
    3. Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    4. Save this episode of data, and continue to next episode of data collection.
    """

    # params from command line args
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False

    # camera names for aloha2
    render_cam_names = [
        "worms_eye_cam",
        "overhead_cam",
        "wrist_cam_left",
        "wrist_cam_right",
    ]
    num_cams = len(render_cam_names)

    # scripted sim policy
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    stat_header = ["episode_idx"]
    if task_name == 'sim_transfer_cube':
        policy_cls = PickAndTransferPolicy
        stat_header += ["cube_x", "cube_y", "cube_z"]
    elif task_name == "sim_pickplace_bottle":
        policy_cls = PickPlaceBottlePolicy
        stat_header += ["bottle_x", "bottle_y", "bottle_z", "bottle_rot"]
    else:
        raise NotImplementedError

    # make dataset_dir and stat table
    success_folder = os.path.join(dataset_dir, "success")
    failed_folder = os.path.join(dataset_dir, "failed")
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(success_folder, exist_ok=True)
        os.makedirs(failed_folder, exist_ok=True)
        with open(os.path.join(success_folder, "stat.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(stat_header)
        with open(os.path.join(failed_folder, "stat.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(stat_header)

    # recording episodes
    print(f"generating {num_episodes} episodes data in total.")
    success = []
    failed_indices = []
    success_episode_idx = 0
    failed_episode_idx = 0
    while success_episode_idx < num_episodes:
        print("======")
        print(f'generating episode {success_episode_idx}/{num_episodes}')
        print('1. Rollout out EE space scripted policy')
        # setup ee environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        episode_rewards = [0]

        # setup plotting
        if onscreen_render:
            fig, axes = plt.subplot_mosaic([['cam0', 'cam1'],
                                            ['cam2', 'cam3'],
                                            ['reward', 'reward']])
            fig.suptitle('EE space rollout')
            plt_imgs = []
            for i, axkey in enumerate(axes):
                axe = axes[axkey]
                if 'cam' in axkey:
                    # cam plot
                    axe.axis('off')
                    plt_img = axe.imshow(ts.observation['images'][render_cam_names[i]])
                else:
                    # reward plot
                    plt_img = axe.plot(episode_rewards)[0]
                    plt.xlim(0, episode_len)
                    plt.ylim(-1, env.task.max_reward+1)
                    plt.title(f"reward (max={env.task.max_reward})")
                    plt.xlabel("simulation steps")
                plt.subplots_adjust(wspace=0, hspace=0)
                plt_imgs.append(plt_img)
            # interactive mode
            plt.ion()

        # start simulation steps
        for step in tqdm(list(range(episode_len))):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            episode_rewards.append(ts.reward)
            if onscreen_render:
                for i, plt_img in enumerate(plt_imgs):
                    if i < num_cams:
                        # update cam plot
                        plt_img.set_data(ts.observation['images'][render_cam_names[i]])
                    else:
                        # update reward plot
                        plt_img.set_xdata(np.arange(len(episode_rewards)))
                        plt_img.set_ydata(episode_rewards)
                plt.pause(0.002)
        if onscreen_render:
            plt.close()

        # calculate episode reward, determine if successful or not
        episode_return = np.sum(episode_rewards)
        episode_max_reward = np.max(episode_rewards)
        episode_final_reward = episode_rewards[-1]
        if episode_max_reward == env.task.max_reward:
            print(f"{success_episode_idx=} Successful")
        else:
            print(f"{failed_episode_idx=} Failed, episode_max_reward={episode_max_reward} out of {env.task.max_reward}")

        print('2. Replace gripper pose with gripper position control input')
        # get qpos observations (length=14, 7 for left arm and 7 for right)
        joint_traj = [ts.observation['qpos'] for ts in episode]

        # replace gripper pose with gripper position control input
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[1])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        # get env state stat (e.g. box pose)
        env_state = episode[0].observation['env_state'].copy()
        if task_name == 'sim_transfer_cube':
            cube_xyz = env_state[:3]
        elif task_name == "sim_pickplace_bottle":
            bottle_xyz = env_state[:3]
            bottle_quat = env_state[3:7]
            bottle_angle = Quaternion(bottle_quat).degrees
        else:
            raise NotImplementedError

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('3. Replaying joint commands')
        env = make_sim_env(task_name)
        ENV_STATE[0] = env_state # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()
        episode_replay = [ts]
        episode_rewards = [0]

        # setup plotting
        if onscreen_render:
            fig, axes = plt.subplot_mosaic([['cam0', 'cam1'],
                                            ['cam2', 'cam3'],
                                            ['reward', 'reward']])
            fig.suptitle('replaying joint commands')
            plt_imgs = []
            for i, axkey in enumerate(axes):
                axe = axes[axkey]
                if 'cam' in axkey:
                    # cam plot
                    axe.axis('off')
                    plt_img = axe.imshow(ts.observation['images'][render_cam_names[i]])
                else:
                    # reward plot
                    plt_img = axe.plot(episode_rewards)[0]
                    plt.xlim(0, episode_len)
                    plt.ylim(-1, env.task.max_reward+1)
                    plt.title(f"reward (max={env.task.max_reward})")
                    plt.xlabel("simulation steps")
                    plt.ylabel("reward value")
                plt.subplots_adjust(wspace=0, hspace=0)
                plt_imgs.append(plt_img)
            # interactive mode
            plt.ion()

        # replay joint commands
        for step in tqdm(list(range(len(joint_traj)))): # note: this will increase episode length by 1
            action = joint_traj[step]
            ts = env.step(action)
            episode_replay.append(ts)
            episode_rewards.append(ts.reward)
            if onscreen_render:
                for i, plt_img in enumerate(plt_imgs):
                    if i < num_cams:
                        # update cam plot
                        plt_img.set_data(ts.observation['images'][render_cam_names[i]])
                    else:
                        # update reward plot
                        plt_img.set_xdata(np.arange(len(episode_rewards)))
                        plt_img.set_ydata(episode_rewards)
                plt.pause(0.002)
        if onscreen_render:
            plt.close()

        # calculate episode reward, determine if successful or not
        episode_return = np.sum(episode_rewards)
        episode_max_reward = np.max(episode_rewards)
        episode_final_reward = episode_rewards[-1]
        success_flag = False
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            success_flag = True
            print(f"{success_episode_idx=} Successful")
            with open(os.path.join(success_folder, "stat.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                row = [success_episode_idx]
                if task_name == 'sim_transfer_cube':
                    writer.writerow(row + [round(coord, 3) for coord in cube_xyz])
                elif task_name == "sim_pickplace_bottle":
                    writer.writerow(row + [round(coord, 3) for coord in bottle_xyz] + [round(bottle_angle, 1)])
                else:
                    raise NotImplementedError
        else:
            success.append(0)
            success_flag = False
            failed_indices.append(failed_episode_idx)
            print(f"{failed_episode_idx=} Failed, episode_max_reward={episode_final_reward} out of {env.task.max_reward}")
            with open(os.path.join(failed_folder, "stat.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                row = [failed_episode_idx]
                if task_name == 'sim_transfer_cube':
                    writer.writerow(row + [round(coord, 3) for coord in cube_xyz])
                elif task_name == "sim_pickplace_bottle":
                    writer.writerow(row + [round(coord, 3) for coord in bottle_xyz] + [round(bottle_angle, 1)])
                else:
                    raise NotImplementedError

        print("4. Saving recorded sim data...")
        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        if success_flag:
            dataset_path = os.path.join(dataset_dir, 'success', f'episode_{success_episode_idx}.hdf5')
            success_episode_idx += 1
        else:
            dataset_path = os.path.join(dataset_dir, 'failed', f'episode_{failed_episode_idx}.hdf5')
            failed_episode_idx += 1
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), compression='gzip', compression_opts=2, )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f"episode data saved as {dataset_path} for {time.time() - t0:.1f} secs.")

    # statistics
    print(f'All data saved to {dataset_dir}')
    print(f'Success: {int(np.sum(success))} / {len(success)}')
    print(f'Failed episode indices: {failed_indices}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))
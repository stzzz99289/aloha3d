import argparse
import os

from aloha.robot_utils import (
    JOINT_NAMES,
    load_yaml_file
)
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

STATE_NAMES = JOINT_NAMES + ['gripper']
BASE_STATE_NAMES = ['linear_vel', 'angular_vel']


def load_hdf5(dataset_dir, dataset_name, is_mobile):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        if is_mobile:
            base_action = root['/base_action'][()]
        else:
            base_action = None
        image_dict = {"color_images": {}, "aligned_depth_images": {}}
        for cam_name in root['/observations/color_images/'].keys():
            image_dict["color_images"][cam_name] = root[f'/observations/color_images/{cam_name}'][()]
        for cam_name in root['/observations/aligned_depth_images/'].keys():
            image_dict["aligned_depth_images"][cam_name] = root[f'/observations/aligned_depth_images/{cam_name}'][()]
        # if compressed:
        #     compress_len = root['/compress_len'][()]

    if compressed:
        color_image_dict = image_dict["color_images"]
        for cam_id, cam_name in enumerate(color_image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = color_image_dict[cam_name]
            image_list = []

            # [:1000] to save memory
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list):
                # image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict["color_images"][cam_name] = image_list

    return qpos, qvel, effort, action, base_action, image_dict


def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    robot_base = args['robot']

    base_path = Path(__file__).resolve().parent.parent / "config"

    config = load_yaml_file('robot', robot_base, base_path).get('robot', {})

    is_mobile = config.get('base', False)

    dt = 1/config.get('fps', 50)

    ismirror = args['ismirror']
    if ismirror:
        dataset_name = f'mirror_episode_{episode_idx}'
    else:
        dataset_name = f'episode_{episode_idx}'

    qpos, _, _, action, base_action, image_dict = load_hdf5(
        dataset_dir, dataset_name, is_mobile)
    print('hdf5 loaded!')
    save_videos(
        image_dict,
        dt,
        video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'),
    )
    visualize_joints(
        qpos,
        action,
        plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'),
        config=config,
    )
    if is_mobile:
        visualize_base(
            base_action,
            plot_path=os.path.join(
                dataset_dir, dataset_name + '_base_action.png'),
        )


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        color_image_dict = video["color_images"]
        aligned_depth_image_dict = video["aligned_depth_images"]

        cam_names = list(color_image_dict.keys())
        all_color_videos = []
        for cam_name in cam_names:
            all_color_videos.append(color_image_dict[cam_name])
        all_color_videos = np.concatenate(
            all_color_videos, axis=2)  # width dimension
        
        # Process depth videos
        all_depth_videos = []
        for cam_name in cam_names:
            depth_video = aligned_depth_image_dict[cam_name]  # (ts, h, w)
            # Normalize to 0-1 range and apply colormap
            if cam_name == "camera_high":
                depth_video_norm = np.clip(depth_video / 2000.0, 0, 1)  # Normalize 0-2m to 0-1 (camera_high uses realsense d455)
            else:
                depth_video_norm = np.clip(depth_video / 500.0, 0, 1)  # Normalize 0-1m to 0-1
            depth_video_colored = np.zeros((*depth_video.shape, 3), dtype=np.uint8)
            for t in range(depth_video.shape[0]):
                depth_frame = (depth_video_norm[t] * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
                depth_video_colored[t] = depth_colored
            all_depth_videos.append(depth_video_colored)
        all_depth_videos = np.concatenate(all_depth_videos, axis=2)  # width dimension

        # Arrange color and depth videos in 2x4 grid
        n_frames = all_color_videos.shape[0]
        h = all_color_videos.shape[1]
        w = all_color_videos.shape[2]
        all_cam_videos = np.zeros((n_frames, h*2, w, 3), dtype=np.uint8)
        all_cam_videos[:, :h, :, :] = all_color_videos
        all_cam_videos[:, h:, :, :] = all_depth_videos

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list,
                     command_list,
                     plot_path=None,
                     ylim=None,
                     label_overwrite=None,
                     config: dict = {},
                     ):

    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)  # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    # h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    leader_robots = {arm['name']: arm for arm in config.get('leader_arms', [])}
    follower_robots = {arm['name']: arm for arm in config.get('follower_arms', [])}

    # Initialize an empty list to store matched suffixes
    valid_suffixes = []

    # Identify valid suffixes from paired robots
    for leader_name in leader_robots.keys():
        # Extract suffix after first underscore
        suffix = leader_name.split('_', 1)[1]
        if f"follower_{suffix}" in follower_robots:
            valid_suffixes.append(suffix)

    # Create all_names based on valid suffixes
    all_names = [
        f"{name}_{suffix}" for suffix in valid_suffixes for name in STATE_NAMES]

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


def visualize_single(efforts_list, label, plot_path=None, ylim=None, label_overwrite=None, config: dict = {}):
    efforts = np.array(efforts_list)  # ts, dim
    num_ts, num_dim = efforts.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    leader_robots = {arm['name']: arm for arm in config.get('leader_arms', [])}
    follower_robots = {arm['name']
        : arm for arm in config.get('follower_arms', [])}

    # Initialize an empty list to store matched suffixes
    valid_suffixes = []

    # Identify valid suffixes from paired robots
    for leader_name in leader_robots.keys():
        # Extract suffix after first underscore
        suffix = leader_name.split('_', 1)[1]
        if f"follower_{suffix}" in follower_robots:
            valid_suffixes.append(suffix)

    # Create all_names based on valid suffixes
    all_names = [
        f"{name}_{suffix}" for suffix in valid_suffixes for name in STATE_NAMES]

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(efforts[:, dim_idx], label=label)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_base(readings, plot_path=None):
    readings = np.array(readings)  # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(
            np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20'
        )
        ax.plot(
            np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10'
        )
        ax.plot(
            np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5'
        )
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title('Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title('dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        action='store',
        type=str,
        help='Dataset dir.',
        required=True,
    )
    parser.add_argument(
        '--episode_idx',
        action='store',
        type=int,
        help='Episode index.',
        required=False,
    )

    parser.add_argument(
        '-r', '--robot',
        required=True,
        help='Specify the robot configuration to use: aloha_solo, aloha_static, or aloha_mobile.'
    )

    parser.add_argument('--ismirror', action='store_true')
    main(vars(parser.parse_args()))

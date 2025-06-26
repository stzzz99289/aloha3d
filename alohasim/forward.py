from dm_control import mujoco
import numpy as np
import os
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from constants import START_ARM_POSE
import glob
from pytransform3d.transformations import plot_transform
from mpl_toolkits.mplot3d import Axes3D
import h5py
import cv2

# initialize sim environment
xml_path = os.path.join(os.path.dirname(__file__), 'mjcf/scene_joint_control.xml')
physics = mujoco.Physics.from_xml_path(xml_path)

def load_hdf5(dataset_dir, dataset_name, is_mobile=False):
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

def aloha_forward_kinematics(arm_pose):
    # set arm pose and compute forward kinematics
    left_arm_pose = arm_pose[:6]
    right_arm_pose = arm_pose[6:]
    with physics.reset_context():
        physics.data.qpos[:6] = left_arm_pose
        physics.data.qpos[8:14] = right_arm_pose

    # return ee pose
    world2ee_transforms = []
    base2ee_transforms = []
    for i in range(2):
        if i == 0:
            arm_name = 'left'
        else:
            arm_name = 'right'

        base_xpos = physics.named.data.xpos[f'{arm_name}/base_link']
        base_xquat = physics.named.data.xquat[f'{arm_name}/base_link']
        ee_xpos = physics.named.data.xpos[f'{arm_name}/gripper_base']
        ee_xquat = physics.named.data.xquat[f'{arm_name}/gripper_base']

        # compute world2ee transformation matrix
        ee2world = np.eye(4)
        ee2world[:3, 3] = ee_xpos
        ee_quat = Quaternion(ee_xquat[0], ee_xquat[1], ee_xquat[2], ee_xquat[3])
        ee2world[:3, :3] = ee_quat.rotation_matrix
        world2ee = np.linalg.inv(ee2world)
        world2ee_transforms.append(world2ee)

        # compute base2ee transformation matrix
        base2world = np.eye(4)
        base2world[:3, 3] = base_xpos
        base_quat = Quaternion(base_xquat[0], base_xquat[1], base_xquat[2], base_xquat[3])
        base2world[:3, :3] = base_quat.rotation_matrix
        base2ee = world2ee @ base2world # remember: order is right to left!!!
        base2ee_transforms.append(base2ee)

    return world2ee_transforms, base2ee_transforms

def visualize_world2cam_poses(data_path):
    world2cam_folder = os.path.join(data_path, "world2cam")
    world2cam_transforms = np.load(os.path.join(world2cam_folder, "world2cam_transformations.npy"))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_transform(ax=ax, A2B=np.eye(4), s=0.1, name='world')
    for i, world2cam_transform in enumerate(world2cam_transforms):
        cam_under_world = np.linalg.inv(world2cam_transform)
        plot_transform(ax=ax, A2B=cam_under_world, s=0.1, name=f'world2cam {i}')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0); ax.set_zlim(0, 1.0)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))
    ax.set_title('World2Cam Transforms')
    plt.show()

def visualize_base2cam_poses(data_path, frame_idx):
    base2ee_dir = os.path.join(data_path, "base2ee")
    ee2cam_dir = os.path.join(data_path, "ee2cam")

    # Get list of base2ee transforms
    base2ee_files = sorted(glob.glob(os.path.join(base2ee_dir, '*.npy')))
    base2ee_transforms = [np.load(f) for f in base2ee_files]

    # Get list of ee2cam transforms
    ee2cam_files = sorted(glob.glob(os.path.join(ee2cam_dir, '*.npy')))
    ee2cam_transforms = [np.load(f) for f in ee2cam_files]

    # Plot base2ee transforms   
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_transform(ax=ax, A2B=np.eye(4), s=0.1, name='base')
    for i, base2ee_transform in enumerate(base2ee_transforms):
        if i == frame_idx:
            plot_transform(ax=ax, A2B=base2ee_transform, s=0.1, name=f'base2ee {i}')
            plot_transform(ax=ax, A2B=np.linalg.inv(base2ee_transform), s=0.1, name=f'base2ee_inv {i}')
            # for j, ee2cam_transform in enumerate(ee2cam_transforms):
            #     plot_transform(ax=ax, A2B=base2ee_transform @ ee2cam_transform, s=0.1, name=f'base2cam {i}-{j}')
            break

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')  
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0); ax.set_zlim(0, 1.0)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))
    ax.set_title('Base2EE Transforms')
    plt.show()

def visualize_cam_frames_relative_to_gripper(data_path):
    # Get list of gripper2cam transforms
    gripper2cam_files = sorted(glob.glob(os.path.join(data_path, 'ee2cam/*.npy')))
    gripper2cam_transforms = [np.load(f) for f in gripper2cam_files]

    # Plot gripper2cam transforms
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot gripper2cam transforms
    plot_transform(ax=ax, A2B=np.eye(4), s=0.1, name='gripper_base')
    for i, transform in enumerate(gripper2cam_transforms):
        cam_frame_relative_to_gripper = np.linalg.inv(transform)
        print(f"method{i} camera center: {cam_frame_relative_to_gripper[:3, 3]}")
        plot_transform(ax=ax, A2B=cam_frame_relative_to_gripper, s=0.1, name=f'Method{i}')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.2, 0.2); ax.set_zlim(-0.2, 0.2)
    ax.set_title('Gripper2Cam Transforms')
    # plt.savefig(os.path.join(data_path, 'gripper2cam_transforms.png'))
    plt.show()

def visualize_arm_poses(left_arm_pose, right_arm_pose):
    # test forward kinematics
    # left_arm_pose = START_ARM_POSE[:6]
    # right_arm_pose = START_ARM_POSE[8:14]
    world2ee_transforms, base2ee_transforms = aloha_forward_kinematics(np.concatenate([left_arm_pose, right_arm_pose]))

    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plot table outline - a rectangle centered at origin
    table_length = 1.21  # Length in x direction
    table_width = 0.76   # Width in y direction
    half_length = table_length / 2
    half_width = table_width / 2
    corners = np.array([
        [-half_length, -half_width, 0],  # Bottom left
        [half_length, -half_width, 0],   # Bottom right 
        [half_length, half_width, 0],    # Top right
        [-half_length, half_width, 0],   # Top left
        [-half_length, -half_width, 0]   # Back to start to close the rectangle
    ])
    ax1.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'k-', linewidth=2, label='Table Outline')
    
    # Plot coordinate frames for both end effectors relative to world frame
    for i, transform in enumerate(world2ee_transforms):
        # Create coordinate frame visualization using pytransform3d
        plot_transform(ax=ax1, A2B=transform, s=0.1, name=f'{"Left" if i==0 else "Right"} World2EE')

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(0, 0.5)
    ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 0.25, 1]))
    ax1.set_title('End Effector Poses Relative to World Frame')
    ax1.legend()
    
    # Plot base2ee transforms  
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    plot_transform(ax=ax2, A2B=np.eye(4), s=0.1, name='Base')
    
    # Plot coordinate frames for both end effectors relative to base frame
    for i, transform in enumerate(base2ee_transforms):
        plot_transform(ax=ax2, A2B=transform, s=0.1, name=f'{"Left" if i==0 else "Right"} Base2EE')

    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_xlim(-0.5, 0.5); ax2.set_ylim(-0.5, 0.5); ax2.set_zlim(0, 0.5)
    ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([1, 1, 0.5, 1]))
    ax2.set_title('End Effector Poses Relative to Base Frame')

    # Set equal aspect ratio for both plots
    ax1.set_box_aspect([1,1,1])
    ax2.set_box_aspect([1,1,1])
    
    plt.show()

def joint_positions_to_ee_poses(data_path, arm_side):
    # Get list of joint position files
    joint_pos_dir = os.path.join(data_path, "joint_positions")
    base2ee_dir = os.path.join(data_path, "base2ee")
    os.makedirs(base2ee_dir, exist_ok=True)
    
    # Get all .npy files in joint_positions directory
    joint_files = sorted([f for f in os.listdir(joint_pos_dir) if f.endswith('.npy')])
    for joint_file in joint_files:
        # Load joint positions
        joint_pos = np.load(os.path.join(joint_pos_dir, joint_file))
        if arm_side == 'left':
            arm_joint_positions = np.concatenate([joint_pos, np.zeros(6)])
        elif arm_side == 'right':
            arm_joint_positions = np.concatenate([np.zeros(6), joint_pos])

        # Convert to base2ee transform using forward kinematics
        world2ee_transforms, base2ee_transforms = aloha_forward_kinematics(arm_joint_positions)
        if arm_side == 'left':
            base2ee = base2ee_transforms[0]
        elif arm_side == 'right':
            base2ee = base2ee_transforms[1]
        
        # Save transform with same filename in base2ee directory
        save_path = os.path.join(base2ee_dir, joint_file)
        np.save(save_path, base2ee)

def dataset_qpos_to_ee_poses(dataset_dir, dataset_name):
    qpos, qvel, effort, action, base_action, image_dict = load_hdf5(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name
    )
    base2ee_transforms = []
    for q in qpos:
        # NOTE: aloha-real saved qpos is first right arm then left arm!!
        right_arm_pose = q[:6]
        left_arm_pose = q[7:13]
        _, base2ee_transform = aloha_forward_kinematics(np.concatenate([left_arm_pose, right_arm_pose]))
        base2ee_transforms.append(base2ee_transform)
    base2ee_transforms = np.array(base2ee_transforms)
    base2ee_transforms = {
        "camera_wrist_left": base2ee_transforms[:, 0, :, :],
        "camera_wrist_right": base2ee_transforms[:, 1, :, :],
    }
    
    # save base2ee transforms
    np.savez(os.path.join(dataset_dir, f'{dataset_name}_base2ee_transforms.npz'), **base2ee_transforms)

if __name__ == "__main__":
    pass

    # joint_positions_to_ee_poses(data_path="/home/tianze/Documents/sim2real2sim/aloha3d/cam_calibration/data/aloha2_lay/left_arm", arm_side='left')
    # joint_positions_to_ee_poses(data_path="/home/tianze/Documents/sim2real2sim/aloha3d/cam_calibration/data/aloha2_lay/right_arm", arm_side='right')
    
    # visualize_cam_frames_relative_to_gripper(data_path='/home/tianze/Documents/sim2real2sim/cam_calibration/data/aloha2_lay/camera_wrist_left')
    # visualize_cam_frames_relative_to_gripper(data_path='/home/tianze/Documents/sim2real2sim/cam_calibration/data/aloha2_lay/camera_wrist_right')
    
    # visualize_base2cam_poses(data_path='/home/tianze/Documents/sim2real2sim/cam_calibration/data/camera_wrist_left', frame_idx=0)

    # visualize_world2cam_poses(data_path='/home/tianze/Documents/sim2real2sim/cam_calibration/data/aloha2_lay/camera_high')
    # visualize_world2cam_poses(data_path='/home/tianze/Documents/sim2real2sim/cam_calibration/data/aloha2_stand/camera_low')

    dataset_qpos_to_ee_poses(
        dataset_dir='/home/tianze/Documents/sim2real2sim/data/head',
        dataset_name='episode_1'
    )
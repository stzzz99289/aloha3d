import numpy as np
import os
from matplotlib import pyplot as plt
from pytransform3d.transformations import plot_transform
from mpl_toolkits.mplot3d import Axes3D

def average_transforms(transforms):
    """Average multiple 4x4 transformation matrices while preserving SE(3) properties.
    
    Args:
        transforms: [N, 4, 4] array of transformation matrices
    Returns:
        [4, 4] averaged transformation matrix
    """
    # Average translation part
    translation = np.mean(transforms[:, :3, 3], axis=0)
    
    # Average rotation part while preserving orthogonality
    U, _, Vh = np.linalg.svd(np.mean(transforms[:, :3, :3], axis=0))
    rotation = U @ Vh
    
    # Combine into transformation matrix
    avg_transform = np.eye(4)
    avg_transform[:3, :3] = rotation
    avg_transform[:3, 3] = translation
    return avg_transform

if __name__ == "__main__":
    result_folder_name = "res"
    result_folder_path = os.path.join(os.path.dirname(__file__), result_folder_name)
    os.makedirs(result_folder_path, exist_ok=True)

    """
    3d sensing results
    """
    world2cam_high_transforms = np.load(os.path.join(
        os.path.dirname(__file__), "data", "aloha2_lay", "camera_high", "world2cam", "world2cam_transformations.npy"
    ))
    world2cam_high = average_transforms(world2cam_high_transforms)

    world2cam_low_transforms = np.load(os.path.join(
        os.path.dirname(__file__), "data", "aloha2_stand", "camera_low", "world2cam", "world2cam_transformations.npy"
    ))
    world2cam_low = average_transforms(world2cam_low_transforms)

    gripper2cam_wrist_left_path = os.path.join(
        os.path.dirname(__file__), "data", "aloha2_lay", "camera_wrist_left", "ee2cam"
    )
    gripper2cam_wrist_left_files = [f for f in os.listdir(gripper2cam_wrist_left_path) if f.endswith('.npy')]
    gripper2cam_wrist_left_transforms = np.stack([
        np.load(os.path.join(gripper2cam_wrist_left_path, f)) 
        for f in gripper2cam_wrist_left_files
    ])
    gripper2cam_wrist_left_transforms = np.delete(gripper2cam_wrist_left_transforms, [3, 6], axis=0)
    gripper2cam_wrist_left = average_transforms(gripper2cam_wrist_left_transforms)
    
    gripper2cam_wrist_right_path = os.path.join(
        os.path.dirname(__file__), "data", "aloha2_lay", "camera_wrist_right", "ee2cam"
    )
    gripper2cam_wrist_right_files = [f for f in os.listdir(gripper2cam_wrist_right_path) if f.endswith('.npy')]
    gripper2cam_wrist_right_transforms = np.stack([
        np.load(os.path.join(gripper2cam_wrist_right_path, f)) 
        for f in gripper2cam_wrist_right_files
    ])
    gripper2cam_wrist_right_transforms = np.delete(gripper2cam_wrist_right_transforms, [3, 6], axis=0)
    gripper2cam_wrist_right = average_transforms(gripper2cam_wrist_right_transforms)

    world2base_left = np.load(os.path.join(
        os.path.dirname(__file__), "data", "aloha2_lay", "camera_wrist_left", "world2base", "method_5.npy"
    ))
    world2base_right = np.load(os.path.join(
        os.path.dirname(__file__), "data", "aloha2_lay", "camera_wrist_right", "world2base", "method_5.npy"
    ))

    np.save(os.path.join(result_folder_path, "world2cam_camera_high.npy"), world2cam_high)
    np.save(os.path.join(result_folder_path, "world2cam_camera_low.npy"), world2cam_low)
    np.save(os.path.join(result_folder_path, "gripper2cam_wrist_left.npy"), gripper2cam_wrist_left)
    np.save(os.path.join(result_folder_path, "gripper2cam_wrist_right.npy"), gripper2cam_wrist_right)
    np.save(os.path.join(result_folder_path, "world2base_left.npy"), world2base_left)
    np.save(os.path.join(result_folder_path, "world2base_right.npy"), world2base_right)

    """
    visualization
    """
    for i in range(10):
        # compute all usefule transforms
        base2ee_left = np.load(os.path.join(
            os.path.dirname(__file__), "data", "aloha2_lay", "left_arm", "base2ee", f"{i:02d}.npy"
        ))
        base2ee_right = np.load(os.path.join(
            os.path.dirname(__file__), "data", "aloha2_lay", "right_arm", "base2ee", f"{i:02d}.npy"
        ))
        world2ee_left = np.dot(base2ee_left, world2base_left)
        world2ee_right = np.dot(base2ee_right, world2base_right)
        world2cam_left = np.dot(gripper2cam_wrist_left, world2ee_left)
        world2cam_right = np.dot(gripper2cam_wrist_right, world2ee_right)
        
        # create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

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
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'k-', linewidth=2, label='Table Outline')

        # plot all frames
        plot_transform(ax=ax, A2B=np.eye(4), s=0.1, name='world frame')
        plot_transform(ax=ax, A2B=np.linalg.inv(world2cam_high), s=0.1, name='camera_high')
        plot_transform(ax=ax, A2B=np.linalg.inv(world2cam_low), s=0.1, name='camera_low')
        plot_transform(ax=ax, A2B=np.linalg.inv(world2cam_left), s=0.1, name='camera_wrist_left')
        plot_transform(ax=ax, A2B=np.linalg.inv(world2cam_right), s=0.1, name='camera_wrist_right')
        plot_transform(ax=ax, A2B=np.linalg.inv(world2base_left), s=0.1, name='left arm base')
        plot_transform(ax=ax, A2B=np.linalg.inv(world2base_right), s=0.1, name='right arm base')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1.0)
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.6, 1.6, 1.0, 1]))
        ax.set_title('Embodied 3D Sensing')
        ax.set_box_aspect([1,1,1])
        
        plt.savefig(os.path.join(result_folder_path, f"sensing_{i:02d}.png"))
        # plt.show()

        
        
    
    

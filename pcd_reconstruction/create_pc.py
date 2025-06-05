import os
import h5py
import numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d
from pyquaternion import Quaternion
from dm_control import mujoco
from utils import load_hdf5

def aligned_images_to_pc(depth_images, color_images, intrinsic_matrix, depth_scale, depth_clip):
    # Get camera intrinsics
    fx = intrinsic_matrix[0,0]
    fy = intrinsic_matrix[1,1]
    cx = intrinsic_matrix[0,2] 
    cy = intrinsic_matrix[1,2]

    # Generate meshgrid of pixel coordinates
    height, width = depth_images[0].shape
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    # Process each frame
    pointclouds = []
    num_frames = len(depth_images)
    for frame_idx in tqdm(range(num_frames), total=num_frames, desc='Processing frames'):
        # Get depth and color for current frame
        depth = depth_images[frame_idx] / depth_scale # depth unit after scaling should be meter
        color = color_images[frame_idx]
        # Make all values outside of clip range as zeros
        depth = np.where(depth > depth_clip, 0, depth)

        # Calculate 3D coordinates
        Z = depth
        X = (x_grid - cx) * Z / fx
        Y = (y_grid - cy) * Z / fy

        # Stack coordinates and reshape to (N,3)
        points = np.stack([X, Y, Z], axis=-1)
        points = points.reshape(-1, 3)

        # Get colors and reshape to (N,3)
        colors = color.reshape(-1, 3)

        # Remove points with zero depth
        valid_points = Z.reshape(-1) > 0
        points = points[valid_points]
        colors = colors[valid_points]
        
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0,1]
        pointclouds.append(pcd)

    return pointclouds

def transform_pointclouds(pointclouds, transformations):
    transformed_pointclouds = []

    if transformations.ndim == 3: # (N, 4, 4) transformations
        assert len(pointclouds) == transformations.shape[0]
        for pointcloud, transformation in zip(pointclouds, transformations):
            transformed_pointcloud = pointcloud.transform(transformation)
            transformed_pointclouds.append(transformed_pointcloud)
    elif transformations.ndim == 2: # (4, 4) transformation
        for pointcloud in pointclouds:
            transformed_pointcloud = pointcloud.transform(transformations)
            transformed_pointclouds.append(transformed_pointcloud)
    else:
        raise ValueError(f"Transformations must be of shape (N, 4, 4) or (4, 4), but got {transformations.shape}")
    
    return transformed_pointclouds

def load_transformations(res_dir):
    transformations = {}
    transformations["world2cam_camera_high"] = np.load(os.path.join(res_dir, "world2cam_camera_high.npy"))
    transformations["world2cam_camera_low"] = np.load(os.path.join(res_dir, "world2cam_camera_low.npy"))
    transformations["gripper2cam_camera_wrist_left"] = np.load(os.path.join(res_dir, "gripper2cam_wrist_left.npy"))
    transformations["gripper2cam_camera_wrist_right"] = np.load(os.path.join(res_dir, "gripper2cam_wrist_right.npy"))
    transformations["world2base_camera_wrist_left"] = np.load(os.path.join(res_dir, "world2base_left.npy"))
    transformations["world2base_camera_wrist_right"] = np.load(os.path.join(res_dir, "world2base_right.npy"))
    return transformations

if __name__ == "__main__":
    # define data path and name
    dataset_dir = '/home/tianze/Documents/sim2real2sim/data/real2sim'
    dataset_name = 'episode_1'
    transformations_dir = '/home/tianze/Documents/sim2real2sim/cam_calibration/res'

    # load video data
    qpos, qvel, effort, action, base_action, image_dict = load_hdf5(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name
    )
    frame_num = len(qpos)

    # load transformation data
    transformations = load_transformations(res_dir=transformations_dir)

    # define camera names
    camera_names = ['camera_wrist_left', 'camera_wrist_right', 'camera_low', 'camera_high']
    eye_in_hand_camera_names = ['camera_wrist_left', 'camera_wrist_right']
    eye_to_hand_camera_names = ['camera_high', 'camera_low']

    # get point clouds under camera frame
    pointclouds_world = {}
    for camera_name in camera_names:
        print(f"Processing {camera_name}...")        

        depth_images = image_dict["aligned_depth_images"][camera_name]
        color_images = image_dict["color_images"][camera_name]
        intrinsic_matrix = np.load(f'/home/tianze/Documents/sim2real2sim/cam_calibration/intrinsics/aloha2/camera_{camera_name}_color_intrinsics.npy')

        if camera_name == 'camera_high': # camera high is d455 model
            depth_clip = 2.0
        else:
            depth_clip = 0.5
        pointclouds_camera = aligned_images_to_pc(depth_images, color_images, intrinsic_matrix, 
                                                  depth_scale=1000.0, depth_clip=depth_clip)
        
        # output_dir = os.path.join(os.path.dirname(__file__), 'pointclouds', dataset_name, camera_name)
        # os.makedirs(output_dir, exist_ok=True)
        # for frame_idx, pcd in tqdm(enumerate(pointclouds_camera), total=len(pointclouds_camera), desc='Saving point clouds under camera frame'):
        #     # Save as PLY file
        #     output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.ply')
        #     o3d.io.write_point_cloud(output_path, pcd)

        if camera_name in eye_to_hand_camera_names:
            world2cam_transformation = transformations[f"world2cam_{camera_name}"]
            cam2world_transformations = np.linalg.inv(world2cam_transformation)
        elif camera_name in eye_in_hand_camera_names:
            gripper2cam_transformation = transformations[f"gripper2cam_{camera_name}"]
            world2base_transformation = transformations[f"world2base_{camera_name}"]
            base2gripper_transformations = np.load(os.path.join(dataset_dir, f'{dataset_name}_base2ee_transforms.npz'))[f'{camera_name}']
            cam2world_transformations = []
            world2cam_transformations = gripper2cam_transformation @ base2gripper_transformations @ world2base_transformation
            cam2world_transformations = np.linalg.inv(world2cam_transformations)

        pointclouds_world[camera_name] = transform_pointclouds(pointclouds_camera, cam2world_transformations)

    # combine world point clouds in each frame
    pointclouds_world_combined = []
    for frame_idx in range(frame_num):
        # Create a new empty point cloud
        combined_pcd = o3d.geometry.PointCloud()
        
        # Add points from each camera's point cloud for this frame
        for camera_name in pointclouds_world.keys():
            # Get the point cloud for this camera and frame
            pcd = pointclouds_world[camera_name][frame_idx]
            
            # Combine points, colors, and normals if available
            combined_pcd.points.extend(pcd.points)
            if pcd.has_colors():
                combined_pcd.colors.extend(pcd.colors)
        
        pointclouds_world_combined.append(combined_pcd)

    # save point clouds under world frame
    for camera_name in pointclouds_world.keys():
        output_dir = os.path.join(os.path.dirname(__file__), 'pointclouds', dataset_name, f'{camera_name}_world')
        os.makedirs(output_dir, exist_ok=True)
        for frame_idx, pcd in tqdm(enumerate(pointclouds_world[camera_name]), total=len(pointclouds_world[camera_name]), desc=f'Saving {camera_name} point clouds under world frame'):
            output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.ply')
            o3d.io.write_point_cloud(output_path, pcd)

    output_dir = os.path.join(os.path.dirname(__file__), 'pointclouds', dataset_name, 'world_combined')
    os.makedirs(output_dir, exist_ok=True)
    for frame_idx, pcd in tqdm(enumerate(pointclouds_world_combined), total=len(pointclouds_world_combined), desc='Saving point clouds under world frame'):
        output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.ply')
        o3d.io.write_point_cloud(output_path, pcd)
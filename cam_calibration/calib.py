import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

# global calibration config
chessboard_position = None
calib_config = None

"""
NOTE: The computation of this function depends on the chessboard position!
TODO: using config file to specify the chessboard position to be used in this function
"""
def set_calib_config(chessboard_position):
    global calib_config
    if chessboard_position == 'lay':
        calib_config = {
            'board_position': 'lay',
            'pattern_size': (6, 9),
            'square_size': 25/1000,
            'data_folder': os.path.join(os.path.dirname(__file__), 'data', 'aloha2_lay'),
            'intrinsic_folder': os.path.join(os.path.dirname(__file__), 'intrinsics', 'aloha2')
        }
    elif chessboard_position == 'stand':
        calib_config = {
            'board_position': 'stand',
            'pattern_size': (8, 11),
            'square_size': 20/1000,
            'data_folder': os.path.join(os.path.dirname(__file__), 'data', 'aloha2_stand'),
            'intrinsic_folder': os.path.join(os.path.dirname(__file__), 'intrinsics', 'aloha2')
        }

def compute_object_points_in_world_frame():
    board_position = calib_config['board_position']
    pattern_size = calib_config['pattern_size']
    square_size = calib_config['square_size']

    if board_position == 'lay':
        x_offset = - (14 - 1.5 - 2.5) / 100
        y_offset = - (18.5 - 2.5 - 2.5) / 100
        z_offset = 1.0 / 100
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32) + z_offset
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)[:, ::-1] * square_size + np.array([x_offset, y_offset])
    elif board_position == 'stand':
        x_offset = - (14 - 3.0 - 2.0) / 100
        y_offset = + (21 - 18.5) / 100
        z_offset = + (1.5 + 2.0) / 100
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32) + y_offset
        object_points[:, 0::2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)[:, ::-1] * square_size + np.array([x_offset, z_offset])
    else:
        raise ValueError(f"Invalid board position: {board_position}")
    
    return object_points

def compute_camera_poses(chessboard_corners, intrinsic_matrix):
    # Create the object points.Object points are points in the real world that we want to find the pose of.
    object_points = compute_object_points_in_world_frame()

    # Estimate the pose of the chessboard corners
    RTarget2Cam = []
    TTarget2Cam = []
    for corners in chessboard_corners:
        # rvec is the rotation vector, tvec is the translation vector from 3d corners to 2d image corners
        _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, None)
        # R is the rotation matrix from the chessboard frame to the camera frame
        R, _ = cv2.Rodrigues(rvec)  
        RTarget2Cam.append(R)
        TTarget2Cam.append(tvec)

    return RTarget2Cam, TTarget2Cam

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, res_save_folder):
    total_error = 0
    num_points = 0
    errors = []

    for i in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
        error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        errors.append(error)
        total_error += error
        num_points += 1

    mean_error = total_error / num_points

    # Plotting the bar graph
    # Create figure with academic-style sizing
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    img_indices = range(1, len(errors) + 1)
    bars = ax.bar(img_indices, errors, 
                    color='#2E86C1',  # Professional blue color
                    edgecolor='black',
                    linewidth=1,
                    alpha=0.8)
    
    # Customize axes
    ax.set_ylim(0, 0.1)
    ax.set_xticks(img_indices)
    ax.set_xlabel('Image Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reprojection Error (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('Reprojection Error Across Calibration Images', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Customize grid and spines
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save both PNG and SVG formats
    fig.savefig(os.path.join(res_save_folder, "reprojection_error.png"), bbox_inches='tight')
    fig.savefig(os.path.join(res_save_folder, "reprojection_error.svg"), bbox_inches='tight', format='svg')

    return mean_error

def calculate_intrinsics(chessboard_corners, available_indices, pattern_size, square_size, image_size, res_save_folder):
    # Find the corners of the chessboard in the image
    imgpoints = chessboard_corners
    # Find the corners of the chessboard in the real world
    objpoints = []
    for i in range(len(available_indices)):
        # objp = compute_object_points_in_world_frame()
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        objpoints.append(objp)
    # Find the intrinsic matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    # Calculate the re-projection error
    reprojection_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, res_save_folder)
    print(f"The average re-projection error from the intrinsics calibration is: {reprojection_error:.2f} pixels")
    return mtx

def find_chessboard_corners(images, pattern_size, res_save_folder):
    chessboard_corners = []
    available_indices = []
    print("finding chessboard corners...")

    detected_corners_folder = os.path.join(res_save_folder, "detected_corners")
    if not os.path.exists(detected_corners_folder):
        os.makedirs(detected_corners_folder)

    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        if ret:
            chessboard_corners.append(corners)
            cv2.drawChessboardCorners(image, pattern_size, corners, ret)
            cv2.imwrite(os.path.join(detected_corners_folder, f"{i:02d}.png"), image)
            available_indices.append(i)
        else:
            print(f"No chessboard found in image {i:02d}")

    return chessboard_corners, available_indices

def calibration_eyeinhand(images, base2ee_transforms, res_save_folder):
    # find chessboard corners and index of images with chessboard corners
    chessboard_corners, available_indices = find_chessboard_corners(images, calib_config['pattern_size'], res_save_folder)
    
    # intrinsics calibration
    intrinsic_matrix = calculate_intrinsics(chessboard_corners, available_indices,
                                            calib_config['pattern_size'], calib_config['square_size'],
                                            images[0].shape[:2], res_save_folder)
    np.save(os.path.join(res_save_folder, "intrinsic_matrix.npy"), intrinsic_matrix)
    
    # calculate transformation: world -> camera
    world2cam_rotations, world2cam_translations = compute_camera_poses(chessboard_corners, intrinsic_matrix)
    world2cam_transforms = [np.concatenate((R, T), axis=1) for R, T in zip(world2cam_rotations, world2cam_translations)]
    for i in range(len(world2cam_transforms)):
        world2cam_transforms[i] = np.concatenate((world2cam_transforms[i], np.array([[0, 0, 0, 1]])), axis=0)

    # calculate transformation: camera -> world
    cam2world_transforms = [np.linalg.inv(T) for T in world2cam_transforms]
    cam2world_rotations = [T[:3, :3] for T in cam2world_transforms]
    cam2world_rotation_vectors = [cv2.Rodrigues(R)[0] for R in cam2world_rotations]
    cam2world_translations = [T[:3, 3] for T in cam2world_transforms]

    # calculate transformation: base -> end-effector
    base2ee_transforms = [base2ee_transforms[i] for i in available_indices]
    base2ee_rotations = [T[:3, :3] for T in base2ee_transforms]
    base2ee_rotation_vectors = [cv2.Rodrigues(R)[0] for R in base2ee_rotations]
    base2ee_translations = [T[:3, 3] for T in base2ee_transforms]

    # calculate transformation: end-effector -> base
    ee2base_transforms = [np.linalg.inv(T) for T in base2ee_transforms]
    ee2base_rotations = [T[:3, :3] for T in ee2base_transforms]
    ee2base_rotation_vectors = [cv2.Rodrigues(R)[0] for R in ee2base_rotations]
    ee2base_translations = [T[:3, 3] for T in ee2base_transforms]

    # calculate transformations: camera -> end-effector
    cam2ee_folder = os.path.join(res_save_folder, "cam2ee")
    ee2cam_folder = os.path.join(res_save_folder, "ee2cam")
    world2base_folder = os.path.join(res_save_folder, "world2base")
    if not os.path.exists(cam2ee_folder):
        os.mkdir(cam2ee_folder)
    if not os.path.exists(ee2cam_folder):
        os.mkdir(ee2cam_folder)
    if not os.path.exists(world2base_folder):
        os.mkdir(world2base_folder)

    for i in range(5):
        print(f"solving hand-eye calibration using method {i}...")
        cam2ee_rotation, cam2ee_translation = cv2.calibrateHandEye(
                R_gripper2base=ee2base_rotations, 
                t_gripper2base=ee2base_translations,
                R_target2cam=world2cam_rotations, 
                t_target2cam=world2cam_translations,
                method=i
            )
        cam2ee_transform = np.concatenate((cam2ee_rotation, cam2ee_translation), axis=1)
        cam2ee_transform = np.concatenate((cam2ee_transform, np.array([[0, 0, 0, 1]])), axis=0)
        np.save(os.path.join(cam2ee_folder, f"method_{i}.npy"), cam2ee_transform)
        ee2cam_transform = np.linalg.inv(cam2ee_transform)
        np.save(os.path.join(ee2cam_folder, f"method_{i}.npy"), ee2cam_transform)

    for i in range(2):
        print(f"solving hand-eye calibration using method {i+5}...")
        base2world_rotation, base2world_translation, ee2cam_rotation, ee2cam_translation = cv2.calibrateRobotWorldHandEye(
                R_world2cam=world2cam_rotations, 
                t_world2cam=world2cam_translations,
                R_base2gripper=base2ee_rotations, 
                t_base2gripper=base2ee_translations,
                method=i
            )
        ee2cam_transform = np.concatenate((ee2cam_rotation, ee2cam_translation), axis=1)
        ee2cam_transform = np.concatenate((ee2cam_transform, np.array([[0, 0, 0, 1]])), axis=0)
        np.save(os.path.join(ee2cam_folder, f"method_{i+5}.npy"), ee2cam_transform)
        cam2ee_transform = np.linalg.inv(ee2cam_transform)
        np.save(os.path.join(cam2ee_folder, f"method_{i+5}.npy"), cam2ee_transform)

        base2world_transform = np.concatenate((base2world_rotation, base2world_translation), axis=1)
        base2world_transform = np.concatenate((base2world_transform, np.array([[0, 0, 0, 1]])), axis=0)
        world2base_transform = np.linalg.inv(base2world_transform)
        np.save(os.path.join(world2base_folder, f"method_{i+5}.npy"), world2base_transform)
        
def calibration_eyetohand(images, intrinsic_path, res_save_folder):
    # find chessboard corners and index of images with chessboard corners
    chessboard_corners, available_indices = find_chessboard_corners(images, calib_config['pattern_size'], res_save_folder)

    # load intrinsic matrix from realsense config
    intrinsic_matrix = np.load(intrinsic_path)
    np.save(os.path.join(res_save_folder, "intrinsic_matrix.npy"), intrinsic_matrix)

    # calculate transformation: world -> camera
    world2cam_rotations, world2cam_translations = compute_camera_poses(chessboard_corners, intrinsic_matrix)
    world2cam_transforms = [np.concatenate((R, T), axis=1) for R, T in zip(world2cam_rotations, world2cam_translations)]
    for i in range(len(world2cam_transforms)):
        world2cam_transforms[i] = np.concatenate((world2cam_transforms[i], np.array([[0, 0, 0, 1]])), axis=0)
    world2cam_rotations = [T[:3, :3] for T in world2cam_transforms]
    world2cam_rotation_vectors = [cv2.Rodrigues(R)[0] for R in world2cam_rotations]
    world2cam_translation_vectors = [T[:3, 3] for T in world2cam_transforms]

    # calculate reprojection error given intrinsic matrix and world2cam transforms
    imgpoints = chessboard_corners
    objpoints = []; rvecs = []; tvecs = []
    for i in available_indices:
        objpoints.append(compute_object_points_in_world_frame())

    reprojection_error = calculate_reprojection_error(objpoints, imgpoints, 
                                                      rvecs=world2cam_rotation_vectors, 
                                                      tvecs=world2cam_translation_vectors, 
                                                      mtx=intrinsic_matrix, dist=None, res_save_folder=res_save_folder)
    print(f"The average re-projection error from the intrinsics calibration is: {reprojection_error:.2f} pixels")

    world2cam_folder = os.path.join(res_save_folder, "world2cam")
    if not os.path.exists(world2cam_folder):
        os.mkdir(world2cam_folder)
    np.save(os.path.join(world2cam_folder, f"world2cam_transformations.npy"), world2cam_transforms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chessboard_position", type=str, default="lay", choices=["lay", "stand"])
    args = parser.parse_args()
    chessboard_position = args.chessboard_position
    set_calib_config(chessboard_position)
    
    # load images data
    camera_names = ['camera_high', 'camera_low', 'camera_wrist_left', 'camera_wrist_right']
    camera_images_dict = {}
    for camera_name in camera_names:
        image_folder = os.path.join(calib_config['data_folder'], camera_name, 'images')
        image_files = sorted(glob.glob(f'{image_folder}/*.png'))
        images = [cv2.imread(f) for f in image_files]
        camera_images_dict[camera_name] = images

    # load base2ee data
    arm_names = ['left_arm', 'right_arm']
    base2ee_dict = {}
    for arm_name in arm_names:
        base2ee_folder = os.path.join(calib_config['data_folder'], arm_name, 'base2ee')
        base2ee_files = sorted(glob.glob(f'{base2ee_folder}/*.npy'))
        base2ee_transforms = [np.load(f) for f in base2ee_files]
        base2ee_dict[arm_name] = base2ee_transforms

    if chessboard_position == 'lay':
        # eye-in-hand calibration
        # TODO: how to use inverse_brown_conrady distortion model in realsense intrinsic when calibrating?
        print("\n\n=== calibrating left wrist camera ===")
        calibration_eyeinhand(images=camera_images_dict['camera_wrist_left'],
                            base2ee_transforms=base2ee_dict['left_arm'], 
                            res_save_folder=os.path.join(calib_config['data_folder'], "camera_wrist_left"))
        print("\n\n=== calibrating right wrist camera ===")
        calibration_eyeinhand(images=camera_images_dict['camera_wrist_right'], 
                                base2ee_transforms=base2ee_dict['right_arm'], 
                                res_save_folder=os.path.join(calib_config['data_folder'], "camera_wrist_right"))

        # fixed-eye calibration
        print("\n\n=== calibrating high camera ===")
        camera_name = "camera_high"
        calibration_eyetohand(images=camera_images_dict[camera_name], 
                                intrinsic_path=os.path.join(calib_config['intrinsic_folder'], f"camera_{camera_name}_color_intrinsics.npy"),
                                res_save_folder=os.path.join(calib_config['data_folder'], camera_name))

    elif chessboard_position == 'stand':
        # fixed-eye calibration
        print("\n\n=== calibrating low camera ===")
        camera_name = "camera_low"
        calibration_eyetohand(images=camera_images_dict[camera_name], 
                                intrinsic_path=os.path.join(calib_config['intrinsic_folder'], f"camera_{camera_name}_color_intrinsics.npy"),
                                res_save_folder=os.path.join(calib_config['data_folder'], camera_name))
    
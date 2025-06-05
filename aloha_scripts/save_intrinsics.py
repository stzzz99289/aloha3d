import pyrealsense2 as rs
import json
import os
import numpy as np

def save_camera_intrinsics(serial_number, camera_name, output_dir='camera_intrinsics'):
    """
    Save camera intrinsics for a RealSense camera with given serial number
    
    Args:
        serial_number (str): Serial number of the RealSense camera
        output_dir (str): Directory to save the intrinsics JSON file
    """
    # Create pipeline and config objects
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the device with the given serial number
    config.enable_device(serial_number)
    
    # Enable color and depth streams
    config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    
    try:
        # Start the pipeline
        pipeline_profile = pipeline.start(config)
        
        # Get the color sensor intrinsics
        color_stream = pipeline_profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Get the depth sensor intrinsics
        depth_stream = pipeline_profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        # Create intrinsics dictionary
        intrinsics_data = {
            'serial_number': serial_number,
            'color': {
                'width': color_intrinsics.width,
                'height': color_intrinsics.height,
                'ppx': color_intrinsics.ppx,
                'ppy': color_intrinsics.ppy,
                'fx': color_intrinsics.fx,
                'fy': color_intrinsics.fy,
                'model': str(color_intrinsics.model),
                'coeffs': color_intrinsics.coeffs
            },
            'depth': {
                'width': depth_intrinsics.width,
                'height': depth_intrinsics.height,
                'ppx': depth_intrinsics.ppx,
                'ppy': depth_intrinsics.ppy,
                'fx': depth_intrinsics.fx,
                'fy': depth_intrinsics.fy,
                'model': str(depth_intrinsics.model),
                'coeffs': depth_intrinsics.coeffs
            }
        }
        # Create intrinsic matrix for color camera in OpenCV format
        color_intrinsic_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Create intrinsic matrix for depth camera in OpenCV format
        depth_intrinsic_matrix = np.array([
            [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
            [0, depth_intrinsics.fy, depth_intrinsics.ppy], 
            [0, 0, 1]
        ])
        
        # Save intrinsic matrices as .npy files
        color_matrix_file = os.path.join(output_dir, f'camera_{camera_name}_color_intrinsics.npy')
        depth_matrix_file = os.path.join(output_dir, f'camera_{camera_name}_depth_intrinsics.npy')
        np.save(color_matrix_file, color_intrinsic_matrix)
        np.save(depth_matrix_file, depth_intrinsic_matrix)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON file
        output_file = os.path.join(output_dir, f'camera_{camera_name}_intrinsics.json')
        with open(output_file, 'w') as f:
            json.dump(intrinsics_data, f, indent=4)
            
        print(f"Saved intrinsics for camera {camera_name} to {output_file}")
            
    finally:
        pipeline.stop()

def main():
    # Camera serial numbers from config
    camera_serials = {
        "camera_high": "213622252685",  # camera_high
        "camera_low": "130322271268",  # camera_low
        "camera_wrist_right": "130322274120",  # camera_wrist_right
        "camera_wrist_left": "130322271656"   # camera_wrist_left
    }
    
    for camera_name, serial in camera_serials.items():
        try:
            save_camera_intrinsics(serial, camera_name)
        except Exception as e:
            print(f"Error saving intrinsics for camera {serial}: {str(e)}")

if __name__ == "__main__":
    main()

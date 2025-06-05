import os
import numpy as np
import open3d as o3d
import glob
import tkinter as tk
from tkinter import Scale, Button, Frame, Label

def load_point_cloud(file_path):
    """Load point cloud data from ply file."""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def visualize_point_clouds(camera_name='camera_high', episode_name='episode_0'):
    """Visualize a series of point clouds with a slider to switch between frames."""
    # Find all point cloud files
    pc_dir = os.path.join(os.path.dirname(__file__), 'pointclouds', episode_name, camera_name)
    pc_files = sorted(glob.glob(os.path.join(pc_dir, 'frame_*.ply')))
    
    if not pc_files:
        print(f"No point cloud files found in {pc_dir}")
        return
    
    # Create Open3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Viewer", width=1024, height=768)
    
    # Load first point cloud
    current_pcd = load_point_cloud(pc_files[0])
    vis.add_geometry(current_pcd)
    
    # Set initial view
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background
    
    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # Add a table outline (similar to the one in aloha2_sim/forward.py)
    table_length = 1.21  # Length in x direction
    table_width = 0.76   # Width in y direction
    half_length = table_length / 2
    half_width = table_width / 2
    
    # Create table outline as line set
    points = np.array([
        [-half_length, -half_width, 0],  # Bottom left
        [half_length, -half_width, 0],   # Bottom right 
        [half_length, half_width, 0],    # Top right
        [-half_length, half_width, 0],   # Top left
        [-half_length, -half_width, 0]   # Back to start to close the rectangle
    ])
    
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    colors = np.array([[0, 0, 0] for _ in range(len(lines))])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Set line width for table outline
    render_option.line_width = 5.0  # Thicker lines for table frame
    
    vis.add_geometry(line_set)
    
    # Update the current frame
    def update_frame(frame_idx):
        nonlocal current_pcd
        
        # Remove current point cloud
        vis.remove_geometry(current_pcd, reset_bounding_box=False)
        
        # Load new point cloud
        current_pcd = load_point_cloud(pc_files[frame_idx])
        
        # Add new point cloud
        vis.add_geometry(current_pcd, reset_bounding_box=False)
        
        # Update visualization
        vis.update_geometry(current_pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # Update frame label
        frame_label.config(text=f"Frame: {frame_idx} / {len(pc_files)-1}")
    
    # Function to save screenshot
    def save_screenshot():
        # Create screenshots directory if it doesn't exist
        screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots', episode_name, camera_name)
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Get current frame index
        frame_idx = slider.get()
        
        # Save screenshot
        screenshot_path = os.path.join(screenshots_dir, f'screenshot_frame_{frame_idx:04d}.png')
        vis.capture_screen_image(screenshot_path, True)
        print(f"Screenshot saved to {screenshot_path}")
    
    # Create Tkinter GUI for slider
    root = tk.Tk()
    root.title("Frame Control")
    root.geometry("400x200")  # Increased height to accommodate the screenshot button
    
    frame = Frame(root)
    frame.pack(pady=10)
    
    # Create label to show current frame
    frame_label = Label(frame, text=f"Frame: 0 / {len(pc_files)-1}")
    frame_label.pack()
    
    # Create slider
    slider = Scale(
        frame, from_=0, to=len(pc_files)-1, 
        orient=tk.HORIZONTAL, length=300,
        command=lambda val: update_frame(int(float(val)))
    )
    slider.pack(pady=10)
    
    # Create buttons for navigation
    btn_frame = Frame(root)
    btn_frame.pack(pady=5)
    
    prev_btn = Button(
        btn_frame, text="Previous", 
        command=lambda: slider.set(max(0, slider.get()-1))
    )
    prev_btn.pack(side=tk.LEFT, padx=5)
    
    next_btn = Button(
        btn_frame, text="Next", 
        command=lambda: slider.set(min(len(pc_files)-1, slider.get()+1))
    )
    next_btn.pack(side=tk.LEFT, padx=5)
    
    # Add screenshot button
    screenshot_btn = Button(
        btn_frame, text="Save Screenshot",
        command=save_screenshot
    )
    screenshot_btn.pack(side=tk.LEFT, padx=5)
    
    # Function to update visualization while keeping the GUI responsive
    def update_vis():
        vis.poll_events()
        vis.update_renderer()
        root.after(10, update_vis)
    
    # Start the update loop
    update_vis()
    
    # Run the GUI
    root.mainloop()
    
    # Clean up
    vis.destroy_window()

if __name__ == "__main__":
    # Default camera name, can be changed as needed
    visualize_point_clouds(camera_name='world_combined', episode_name='episode_0')

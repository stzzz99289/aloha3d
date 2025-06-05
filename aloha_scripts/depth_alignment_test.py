#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters
import argparse

class DepthAlignmentTest(Node):
    def __init__(self, camera_name):
        super().__init__('depth_alignment_test')
        self.bridge = CvBridge()
        self.camera_name = camera_name
        self.max_working_distance = 20.0 if camera_name == 'camera_high' else 0.5
        self.max_visualize_distance = 2.0 if camera_name == 'camera_high' else 0.5
        self.depth_scale = 1000.0
        
        # Create synchronized subscribers for color and depth images
        if camera_name == 'camera_high':
            color_topic = f'/{camera_name}/camera/color/image_raw'
            depth_topic = f'/{camera_name}/camera/aligned_depth_to_color/image_raw'
        else:
            color_topic = f'/{camera_name}/camera/color/image_rect_raw'
            depth_topic = f'/{camera_name}/camera/aligned_depth_to_color/image_raw'
        color_sub = message_filters.Subscriber(self, Image, color_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        # Synchronize the messages with a time tolerance of 0.1 seconds
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.callback)
        
        self.get_logger().info(f"Subscribed to color topic: {color_topic}")
        self.get_logger().info(f"Subscribed to depth topic: {depth_topic}")
        # self.get_logger().info(f"Subscribed to color and depth topics for camera: {camera_name}")
        
        # Add message counter
        self.msg_count = 0
        
    def callback(self, color_msg, depth_msg):
        try:
            self.msg_count += 1
            if self.msg_count % 30 == 0:  # Log every 30 messages
                self.get_logger().info(f"Received {self.msg_count} pairs of messages")
            
            # Convert ROS Image messages to OpenCV images
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            # Add debug info for depth values
            if self.msg_count % 30 == 0:
                self.get_logger().info(f"Depth image stats - Min: {np.min(depth_image)}, Max: {np.max(depth_image)}, Median: {np.median(depth_image)}")
            
            # Normalize depth image for visualization (0-255)
            # Clip depth values to max working distance (convert from meters to mm)
            max_depth_mm = self.max_working_distance * self.depth_scale
            max_visualize_distance_mm = self.max_visualize_distance * self.depth_scale
            clipped_depth = np.clip(depth_image, 0, max_depth_mm)
            
            # Scale for visualization (0-255)
            scale_factor = 255.0 / max_visualize_distance_mm if max_visualize_distance_mm > 0 else 0.03
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(clipped_depth, alpha=scale_factor), cv2.COLORMAP_HOT)
            
            # Create a blended image to show alignment
            alpha = 0.4  # Transparency factor
            blended_image = cv2.addWeighted(color_image, 1 - alpha, depth_colormap, alpha, 0)
            
            # Create a side-by-side display
            combined_image = np.hstack((color_image, depth_colormap, blended_image))
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            h, w = color_image.shape[:2]
            cv2.putText(combined_image, "Color", (w//2 - 50, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(combined_image, "Depth", (w + w//2 - 50, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(combined_image, "Blended", (2*w + w//2 - 70, 30), font, 1, (255, 255, 255), 2)
            
            # Display the images
            cv2.imshow(f"Depth Alignment Test - {self.camera_name}", combined_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error processing images: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test depth and color alignment for RealSense cameras')
    parser.add_argument('--camera', type=str, default='camera_high', help='camera name')
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        test = DepthAlignmentTest(args.camera)
        rclpy.spin(test)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

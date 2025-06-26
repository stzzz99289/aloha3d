import mujoco
import mujoco.viewer
import numpy as np
import os
import pyquaternion as pyq
import time
from constants import START_ARM_POSE

control_index = 0

def key_callback_data(key, data):
    """
    Callback for key presses but with data passed in
    :param key: Key pressed
    :param data:  MjData object
    :return: None
    """
    global control_index
    if key == 76: # press L, switch control to left arm
        control_index = 0
    elif key == 82: # press R, switch control to right arm
        control_index = 1

    if key == 265:  # Up arrow, positive z movement
        data.mocap_pos[control_index, 2] += 0.01
    elif key == 264:  # Down arrow, negative z movement
        data.mocap_pos[control_index, 2] -= 0.01
    elif key == 263:  # Left arrow, negative x movement
        data.mocap_pos[control_index, 0] -= 0.01
    elif key == 262:  # Right arrow, positive x movement
        data.mocap_pos[control_index, 0] += 0.01
    elif key == 320:  # Numpad 0, positive y movement
        data.mocap_pos[control_index, 1] += 0.01
    elif key == 330:  # Numpad ., negative y movement
        data.mocap_pos[control_index, 1] -= 0.01
    elif key == 260:  # Insert, positive rotation around x axis for 10 degrees
        data.mocap_quat[control_index] = rotate_quaternion(data.mocap_quat[control_index], [1, 0, 0], 10)
    elif key == 261:  # Home, negative rotation around x axis for 10 degrees
        data.mocap_quat[control_index] = rotate_quaternion(data.mocap_quat[control_index], [1, 0, 0], -10)
    elif key == 268:  # Delete, positive rotation around y axis for 10 degrees
        data.mocap_quat[control_index] = rotate_quaternion(data.mocap_quat[control_index], [0, 1, 0], 10)
    elif key == 269:  # End, negative rotation around y axis for 10 degrees
        data.mocap_quat[control_index] = rotate_quaternion(data.mocap_quat[control_index], [0, 1, 0], -10)
    elif key == 266:  # Page Up, positive rotation around z axis for 10 degrees
        data.mocap_quat[control_index] = rotate_quaternion(data.mocap_quat[control_index], [0, 0, 1], 10)
    elif key == 267:  # Page Down, negative rotation around z axis for 10 degrees
        data.mocap_quat[control_index] = rotate_quaternion(data.mocap_quat[control_index], [0, 0, 1], -10)
    else:
        print(f"Unhandled key: {key}")

def rotate_quaternion(quat, axis, angle):
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)

    return q.elements

def main():
    # load aloha scene with mocap control
    xml_path = os.path.join(os.path.dirname(__file__), 'mjcf/scene_mocap.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # set initial states of arms and mocaps
    data.qpos[:16] = START_ARM_POSE
    data.mocap_pos[:2] = np.array([[-0.31718881 + 0.1, -0.0675, 0.31525084],
                                   [0.31718881 - 0.1, -0.0675, 0.31525084]])
    data.mocap_quat[:2] = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])

    def key_callback(key):
        key_callback_data(key, data)

    simulation_fps = 50
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / simulation_fps)

if __name__ == "__main__":
    main()

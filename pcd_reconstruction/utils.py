import h5py
import cv2
import os
from tqdm import tqdm

def save_separated_images(image_dict, output_dir, sample_rate=1):
    color_images = image_dict["color_images"]
    print("saving images to ", output_dir)
    for cam_name, images in tqdm(color_images.items(), total=len(color_images), desc="Saving color images"):
        os.makedirs(os.path.join(output_dir, f"{cam_name}_color_images"), exist_ok=True)
        for frame_idx, image in enumerate(images):
            if frame_idx % sample_rate != 0:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{cam_name}_color_images", f'{frame_idx:05d}.jpg'), image)

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
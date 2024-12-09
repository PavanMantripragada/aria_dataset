import os
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from projectaria_tools.projects import ase
from readers import read_trajectory_file, read_language_file, read_points_file
from interpreter import language_to_bboxes
from projectaria_tools.utils.calibration_utils import rotate_upright_image_and_calibration
from projectaria_tools.core import calibration


UNIT_CUBE_VERTICES = (
    np.array(
        [
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        ]
    )
    * 0.5
)

def transform_3d_points(transform, points):
    N = len(points)
    points_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    transformed_points_h = (transform @ points_h.T).T
    transformed_points = transformed_points_h[:, :-1]
    return transformed_points

class ToTensor(object):
    def __call__(self, sample):
        # converts image shape from (H,W,C) to (C,H,W)
        sample['rgb'] = torch.tensor(sample['rgb'].transpose(0,3,1,2) / 255.0, dtype=torch.float32)
        sample['depth'] = torch.tensor(sample['depth'], dtype=torch.float32)
        sample['extrinsics'] = torch.tensor(sample['extrinsics'], dtype=torch.float32)
        
        return sample

class Rectify(object):
    def __init__(self,undistort=True):
        self.undistort = undistort
        self.device = ase.get_ase_rgb_calibration()

        self.pinhole = calibration.get_linear_camera_calibration(
        self.device.get_image_size()[0],
        self.device.get_image_size()[1],
        self.device.get_focal_lengths()[0])

    def __call__(self, sample):
        undistorted_image = []
        for i in range(sample['rgb'].shape[0]):
            undistorted_image.append(calibration.distort_by_calibration(sample['rgb'][i], self.pinhole, self.device))
        undistorted_image = np.array(undistorted_image).transpose(1, 2, 3, 0)
        sample['rgb'] = np.rot90(undistorted_image,k=3).transpose(3, 0, 1, 2)
        
        undistorted_image = []
        for i in range(sample['depth'].shape[0]): 
            undistorted_image.append(calibration.distort_by_calibration(sample['depth'][i], self.pinhole, self.device))
        undistorted_image = np.array(undistorted_image).transpose(1, 2, 0)
        sample["depth"] = np.rot90(undistorted_image,k=3).transpose(2,0,1)
        sample['depth'] = np.ascontiguousarray(sample['depth'])

        return sample


class SequenceDataset(Dataset):
    def __init__(self, root_dir, scene_count, sequence_length, transform=None):
        self.root_dir = Path(root_dir)
        print("Chosen ASE data path: ", self.root_dir)
        self.scene_count = scene_count
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = self._get_sequences()
        # Load camera calibration
        self.get_mask()
        self.calib = calibration.rotate_camera_calib_cw90deg(self.pinhole)
        self.scale = 0.0649402787400406

        
    def _get_sequences(self):
        sequences = []
        # print(self.scene_count)
        for scene_id in range(self.scene_count):
            scene_path = self.root_dir / str(scene_id)
            rgb_dir = scene_path / "rgb"
            num_frames = len(list(rgb_dir.glob("*.jpg")))
            # print(num_frames)
            start_ids = np.arange(0,num_frames,self.sequence_length)[:-1]
            end_ids = np.minimum(start_ids + self.sequence_length, num_frames)
            # print(start_ids,end_ids)
            sequence = np.column_stack((np.full_like(start_ids, scene_id), start_ids, end_ids)).tolist()
            sequences+= sequence

        # print(sequences)
        # exit()
        return sequences

    def get_mask(self):
        device = ase.get_ase_rgb_calibration()

        pinhole = calibration.get_linear_camera_calibration(
        device.get_image_size()[0],
        device.get_image_size()[1],
        device.get_focal_lengths()[0])


        mask = Image.open("/home/pavan/Documents/projectaria_sandbox/projectaria_tools/data/vignetting_mask/rgb_vignetting_half_reso_mask.jpg")
        width, height = mask.size
        new_width = width // 2
        new_height = height // 2
        mask = mask.resize((new_width, new_height))
        undistorted_mask = calibration.distort_by_calibration(mask, pinhole, device)
        undistorted_mask = np.array(undistorted_mask,dtype=np.float32)/255
        undistorted_mask[undistorted_mask > 0] = 1.0

        self.mask = torch.tensor(undistorted_mask, dtype=torch.float32)
        self.pinhole = pinhole

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        scene_id, start_id, end_id = self.sequences[idx]

        scene_path = self.root_dir / str(scene_id)
        rgb_dir = scene_path / "rgb"
        depth_dir = scene_path / "depth"

        # Load the trajectory using read_trajectory_file() 
        trajectory_path = scene_path / "trajectory.csv"
        trajectory = read_trajectory_file(trajectory_path)

        # Load a scene command language using read_language_file()
        language_path = scene_path / "ase_scene_language.txt"
        entities = read_language_file(language_path)
        entity_boxes = language_to_bboxes(entities)
        corners = []
        for box in entity_boxes:
            box_verts = UNIT_CUBE_VERTICES * box["scale"]
            box_verts = (box["rotation"] @ box_verts.T).T
            box_verts = box_verts + box["center"]
            corners.append(box_verts)
        corners = np.vstack(corners)
        ll = np.min(corners,axis=0)
        ul = np.max(corners,axis=0)
        room_center = (ul+ll)/2
    
        rgb_images = []
        depth_images = []
        extrinsics = []

        for frame_idx in range(start_id,end_id):
            frame_id = str(frame_idx).zfill(7)

            # T_world_from_device
            extrinsics.append(trajectory["Ts_world_from_device"][frame_idx])

            rgb_path = rgb_dir / f"vignette{frame_id}.jpg"
            depth_path = depth_dir / f"depth{frame_id}.png"

            # Load RGB images
            rgb_images.append(Image.open(rgb_path))
            # Load depth images
            depth_images.append(Image.open(depth_path))
        
        # Scale to [-1,1] box centered at room
        depth_images = self.scale*np.array(depth_images).astype(np.float32)
        extrinsics = np.array(extrinsics)
        extrinsics[:,:3,3] = self.scale*(extrinsics[:,:3,3] - np.tile(room_center, (extrinsics.shape[0], 1)))

        sample = {
                'rgb': np.array(rgb_images),
                'depth': depth_images,
                'extrinsics': extrinsics
                }        
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Example usage
if __name__ == "__main__":
    root_dir = "/home/pavan/Documents/projectaria_sandbox/projectaria_tools_ase_data"
    sequence_length = 5
    scene_count = 2
    transform = transforms.Compose([Rectify(),
                                    ToTensor()
                                     ])

    dataset = SequenceDataset(root_dir,scene_count, sequence_length, transform=transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch in dataloader:
        print("RGB images shape:", batch['rgb'].shape)  # (B, N, C, H, W)
        print("Depth images shape:", batch['depth'].shape)  # (B, N, H, W)
        print("Extrinsics shape:", batch['extrinsics'].shape)  # (B,N,4,4)

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(3, 1), dpi=300)
        axes[0].imshow(batch['rgb'][0,0,:,:,:].numpy().transpose(1,2,0))
        axes[0].set_title("RGB Image")
        axes[1].imshow(np.array(batch['depth'][0,0,:,:]), cmap="plasma")
        axes[1].set_title("Metric Depth (mm)")
        for ax in axes:
            ax.axis("off")
        plt.show()
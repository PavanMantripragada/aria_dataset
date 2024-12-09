import os
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from projectaria_tools.projects import ase
device = ase.get_ase_rgb_calibration()


from projectaria_tools.core import data_provider, calibration
pinhole = calibration.get_linear_camera_calibration(
    device.get_image_size()[0],
    device.get_image_size()[1],
    device.get_focal_lengths()[0])


rgb = Image.open("/home/pavan/Documents/projectaria_sandbox/projectaria_tools_ase_data/18/rgb/vignette0000000.jpg")

raw_image = np.array(rgb)
undistorted_image = calibration.distort_by_calibration(raw_image, pinhole, device)



mask = Image.open("/home/pavan/Documents/projectaria_sandbox/projectaria_tools/data/vignetting_mask/rgb_vignetting_half_reso_mask.jpg")
width, height = mask.size
new_width = width // 2
new_height = height // 2
mask = mask.resize((new_width, new_height))
undistorted_mask = calibration.distort_by_calibration(mask, pinhole, device)
undistorted_mask = np.array(undistorted_mask,dtype=np.float32)/255
undistorted_mask[undistorted_mask > 0] = 1.0
# Plot the images
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5), dpi=300)
axes[0].imshow(raw_image)
axes[0].set_title("RGB Image")
axes[1].imshow(undistorted_image) # , cmap="plasma"
axes[1].set_title("Metric Depth (mm)")
axes[2].imshow(undistorted_mask)
axes[2].set_title("Instance Map")
for ax in axes:
    ax.axis("off")
plt.show()

import os
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from projectaria_tools.projects import ase
from readers import read_trajectory_file, read_language_file, read_points_file
from interpreter import language_to_bboxes

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

root_dir = Path("/home/pavan/Documents/projectaria_sandbox/projectaria_tools_ase_data")
max_extent = []
for scene_id in range(500):
    scene_path = root_dir / str(scene_id)

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
    max_extent.append(np.max(ul-ll))

max_extent = np.max(max_extent)
print(max_extent)
scale = 2.0/max_extent
print(scale)
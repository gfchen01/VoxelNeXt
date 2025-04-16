import open3d as o3d
from pathlib import Path
import numpy as np
import os

def read_ply(filename):
        """Custom read ply. Points: [x, y, z, id] = [float, float, float, int]

        Args:
            filename (_type_): _description_

        Returns:
            _type_: _description_
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Parse the header
        header_ended = False
        points = []
        ids = []
        for line in lines:
            if header_ended:
                # Parse the point data
                parts = line.strip().split()
                x, y, z = map(float, parts[:3])
                index = int(parts[3])
                points.append([x, y, z])
                ids.append(index)
            elif line.strip() == 'end_header':
                header_ended = True
        
        # Convert to numpy arrays
        points = np.array(points)
        ids = np.array(ids)
        return points, ids

def get_lidar(idx):
    lidar_file = Path('/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/training') / 'velodyne' / ('%s' % idx)
    # print(str(lidar_file))
    assert lidar_file.exists()
    # mesh = o3d.io.read_triangle_mesh(str(lidar_file))
    # mesh = np.asarray(mesh.vertices)
    mesh, ids = read_ply(str(lidar_file))
    return np.hstack([mesh, np.zeros((mesh.shape[0], 1))])

x_min = 100000
x_max = -100000
y_min = 100000
y_max = -100000
z_min = 100000
z_max = -100000
for file_ in os.listdir('/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/training/velodyne'):
    # file_ = '000513.ply'
    lidar_points = get_lidar(file_)
    # print(len(lidar_points))
    
    if np.max(lidar_points[:, 0]) > x_max:
        x_max = np.max(lidar_points[:, 0])
    if np.min(lidar_points[:, 0]) < x_min:
        x_min = np.min(lidar_points[:, 0])
        
    if np.max(lidar_points[:, 1]) > y_max:
        y_max = np.max(lidar_points[:, 1])
    if np.min(lidar_points[:, 1]) < y_min:
        y_min = np.min(lidar_points[:, 1])
        
    if np.max(lidar_points[:, 2]) > z_max:
        z_max = np.max(lidar_points[:, 2])
    if np.min(lidar_points[:, 2]) < z_min:
        z_min = np.min(lidar_points[:, 2])
        
print([x_min, y_min, z_min, x_max, y_max, z_max])
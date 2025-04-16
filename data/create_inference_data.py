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
    points, ids = read_ply(str(lidar_file))
    
    # find unique ids
    unique_ids = np.unique(ids)
    random_id = np.random.choice(unique_ids)
    points = points[ids == random_id]
    
    print(f'random_id: {idx}:{random_id}')
    
    return points

inferece_pc_path = '/home/luke/NREC/obj_detection/VoxelNeXt/data/inference/velodyne'
if not os.path.exists(inferece_pc_path):
    os.makedirs(inferece_pc_path)
    
for file_ in os.listdir('/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/training/velodyne'):
    # file_ = '000513.ply'
    obj_lidar_points = get_lidar(file_)
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_lidar_points)
    o3d.io.write_point_cloud(inferece_pc_path + '/' + file_, obj_pcd, write_ascii=True)


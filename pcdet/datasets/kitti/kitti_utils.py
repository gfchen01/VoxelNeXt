import numpy as np
from ...utils import box_utils


def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2

def scan2pixels(laserCloud):
  # project scan points to image pixels
  # https://github.com/jizhang-cmu/cmu_vla_challenge_unity/blob/noetic/src/semantic_scan_generation/src/semanticScanGeneration.cpp
  
  # Input: 
  # [#points, 3], x-y-z coordinates of lidar points
  
  # Output: 
  #    point_pixel_idx['horiPixelID'] : horizontal pixel index in the image coordinate
  #    point_pixel_idx['vertPixelID'] : vertical pixel index in the image coordinate
  
  
  L2C_PARA= {"x": 0, "y": 0, "z": 0.235, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963} #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
  CAMERA_PARA= {"hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
  LIDAR_PARA= {"hfov": 360, "vfov": 30}   
  
  laserPixel=[]
  #---------
  # current robot coordinate, set camera as orign, 
  # transform current lidar points from lidar-orign coordinate to camera-orign coordinate
  
  lidarX = 0 #   lidarXStack[imageIDPointer]
  lidarY = 0 # idarYStack[imageIDPointer]
  lidarZ = L2C_PARA["z"] # lidarZStack[imageIDPointer]
  lidarRoll = -L2C_PARA["roll"] #  lidarRollStack[imageIDPointer]
  lidarPitch = -L2C_PARA["pitch"] # lidarPitchStack[imageIDPointer]
  lidarYaw = -L2C_PARA["yaw"]# lidarYawStack[imageIDPointer]

  imageWidth = CAMERA_PARA["width"]
  imageHeight = CAMERA_PARA["height"]
  cameraOffsetZ= 0   #  additional pixel offset due to image cropping? 
  vertPixelOffset=0 #  additional vertical pixel offset due to image cropping

  sinLidarRoll = np.sin(lidarRoll*np.pi / 180.)
  cosLidarRoll = np.cos(lidarRoll*np.pi / 180.)
  sinLidarPitch = np.sin(lidarPitch*np.pi / 180.)
  cosLidarPitch = np.cos(lidarPitch*np.pi / 180.)
  sinLidarYaw = np.sin(lidarYaw*np.pi / 180.)
  cosLidarYaw = np.cos(lidarYaw*np.pi / 180.)
  
  lidar_offset = np.array([lidarX, lidarY, lidarZ])
  camera_offset = np.array([0, 0, cameraOffsetZ])
  
  cloud = laserCloud[:, :3] - lidar_offset
  R_z = np.array([[cosLidarYaw, -sinLidarYaw, 0], [sinLidarYaw, cosLidarYaw, 0], [0, 0, 1]])
  R_y = np.array([[cosLidarPitch, 0, sinLidarPitch], [0, 1, 0], [-sinLidarPitch, 0, cosLidarPitch]])
  R_x = np.array([[1, 0, 0], [0, cosLidarRoll, -sinLidarRoll], [0, sinLidarRoll, cosLidarRoll]])
  cloud = cloud @ R_z @ R_y @ R_x
  cloud = cloud - camera_offset
  
  horiDis = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
  horiPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 1], cloud[:, 0]) + imageWidth / 2 + 1).astype(int) - 1
  vertPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 2], horiDis) + imageHeight / 2 + 1 + vertPixelOffset).astype(int)
  PixelDepth = horiDis
      
#   x1 = laserCloud[:,0] - lidarX
#   y1 = laserCloud[:,1] - lidarY
#   z1 = laserCloud[:,2] - lidarZ

#   x2 = x1 * cosLidarYaw + y1 * sinLidarYaw
#   y2 = -x1 * sinLidarYaw + y1 * cosLidarYaw
#   z2 = z1

#   x3 = x2 * cosLidarPitch - z2 * sinLidarPitch
#   y3 = y2
#   z3 = x2 * sinLidarPitch + z2 * cosLidarPitch

#   x4 = x3
#   y4 = y3 * cosLidarRoll + z3 * sinLidarRoll
#   z4 = -y3 * sinLidarRoll + z3 * cosLidarRoll - cameraOffsetZ

#   horiDis = np.sqrt(x4 * x4 + y4 * y4)
#   horiPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(y4, x4) + imageWidth / 2 + 1).astype(int)-1
#   vertPixelID = (-imageWidth / (2 * np.pi) * np.arctan(z4 / horiDis) + imageHeight / 2 + 1+vertPixelOffset).astype(int)
#   PixelDepth= horiDis
  
  point_pixel_idx={'horiPixelID': horiPixelID, 'vertPixelID': vertPixelID, 'PixelDepth': PixelDepth}
  
  return point_pixel_idx
import os
import shutil

UNBUILT_DATASET = '/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4_unbuilt/'
DATASET_TO_BUILD = '/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/'

if not os.path.exists(DATASET_TO_BUILD + 'training'):
    os.mkdir(DATASET_TO_BUILD + 'training')
    os.mkdir(DATASET_TO_BUILD + 'training/' + 'calib')
    os.mkdir(DATASET_TO_BUILD + 'training/' + 'image_2')
    os.mkdir(DATASET_TO_BUILD + 'training/' + 'label_2')
    os.mkdir(DATASET_TO_BUILD + 'training/' + 'planes')
    os.mkdir(DATASET_TO_BUILD + 'training/' + 'velodyne')
    os.mkdir(DATASET_TO_BUILD + 'training/' + 'velodyne_world')
    
    
if not os.path.exists(DATASET_TO_BUILD + 'testing'):
    os.mkdir(DATASET_TO_BUILD + 'testing')
    os.mkdir(DATASET_TO_BUILD + 'testing/' + 'calib')
    os.mkdir(DATASET_TO_BUILD + 'testing/' + 'image_2')
    os.mkdir(DATASET_TO_BUILD + 'testing/' + 'label_2')
    os.mkdir(DATASET_TO_BUILD + 'testing/' + 'planes')
    os.mkdir(DATASET_TO_BUILD + 'testing/' + 'velodyne')
    os.mkdir(DATASET_TO_BUILD + 'testing/' + 'velodyne_world')

current_file_number_to_save = -1
env_change_idx = {}
prev_folder = None
for folder in os.listdir(UNBUILT_DATASET):
    curr_folder = UNBUILT_DATASET + folder +'/data'
    total_files_in_folders = len(os.listdir(UNBUILT_DATASET + folder + '/data/calib'))
    if folder not in env_change_idx:
        print(folder)
        env_change_idx[folder] = [current_file_number_to_save + 1]
        if prev_folder != None:
            env_change_idx[prev_folder].append(current_file_number_to_save)
    prev_folder = folder
    # env_change_idx.append(current_file_number_to_save+1)
    for current_file_number in range(0, total_files_in_folders):
        current_file_number_to_save += 1
        
        os.symlink(UNBUILT_DATASET + folder +'/data/calib/' + '{0:06d}'.format(current_file_number)+'.txt', DATASET_TO_BUILD + 'training' +'/calib/' + '{0:06d}'.format(current_file_number_to_save)+'.txt')
        os.symlink(UNBUILT_DATASET + folder +'/data/calib/' + '{0:06d}'.format(current_file_number)+'.txt', DATASET_TO_BUILD + 'testing' +'/calib/' + '{0:06d}'.format(current_file_number_to_save)+'.txt')
        
        os.symlink(UNBUILT_DATASET + folder +'/data/label_2/' + '{0:06d}'.format(current_file_number)+'.txt', DATASET_TO_BUILD + 'training' +'/label_2/' + '{0:06d}'.format(current_file_number_to_save)+'.txt')
        os.symlink(UNBUILT_DATASET + folder +'/data/label_2/' + '{0:06d}'.format(current_file_number)+'.txt', DATASET_TO_BUILD + 'testing' +'/label_2/' + '{0:06d}'.format(current_file_number_to_save)+'.txt')
        
        os.symlink(UNBUILT_DATASET + folder +'/data/image_2/' + '{0:06d}'.format(current_file_number)+'.png', DATASET_TO_BUILD + 'training' +'/image_2/' + '{0:06d}'.format(current_file_number_to_save)+'.png')
        os.symlink(UNBUILT_DATASET + folder +'/data/image_2/' + '{0:06d}'.format(current_file_number)+'.png', DATASET_TO_BUILD + 'testing' +'/image_2/' + '{0:06d}'.format(current_file_number_to_save)+'.png')
        
        os.symlink(UNBUILT_DATASET + folder +'/data/velodyne/' + '{0:06d}'.format(current_file_number)+'.ply', DATASET_TO_BUILD + 'training' +'/velodyne/' + '{0:06d}'.format(current_file_number_to_save)+'.ply')
        os.symlink(UNBUILT_DATASET + folder +'/data/velodyne/' + '{0:06d}'.format(current_file_number)+'.ply', DATASET_TO_BUILD + 'testing' +'/velodyne/' + '{0:06d}'.format(current_file_number_to_save)+'.ply')
        
        os.symlink(UNBUILT_DATASET + folder +'/data/velodyne_world/' + '{0:06d}'.format(current_file_number)+'.ply', DATASET_TO_BUILD + 'training' +'/velodyne_world/' + '{0:06d}'.format(current_file_number_to_save)+'.ply')
        os.symlink(UNBUILT_DATASET + folder +'/data/velodyne_world/' + '{0:06d}'.format(current_file_number)+'.ply', DATASET_TO_BUILD + 'testing' +'/velodyne_world/' + '{0:06d}'.format(current_file_number_to_save)+'.ply')

# env_change_idx.append(current_file_number_to_save+1)    
env_change_idx[prev_folder].append(current_file_number_to_save)    
print(env_change_idx)
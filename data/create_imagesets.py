import os
import random

# Open a file in write mode
file_train = open("/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/ImageSets/train.txt", "w")
file_test = open("/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/ImageSets/test.txt", "w")
file_val = open("/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/ImageSets/val.txt", "w")
file_train_val = open("/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/ImageSets/trainval.txt", "w")

for i in range(0, len(os.listdir('/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/training/image_2/'))):
    randnum = random.uniform(0.0, 1.0)
    if randnum < 0.80:
        file_train.write("{:06d}".format(i)  + "\n")
        file_train_val.write("{:06d}".format(i)  + "\n")
    # elif randnum >=0.6 and randnum < 0.8:
        # file_val.write("{:06d}".format(i)  + "\n")
        # file_train_val.write("{:06d}".format(i)  + "\n")
    elif randnum >=0.8 and randnum < 1.0:
        file_val.write("{:06d}".format(i)  + "\n")
        file_train_val.write("{:06d}".format(i)  + "\n")
        file_test.write("{:06d}".format(i)  + "\n")
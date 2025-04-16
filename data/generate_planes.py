import os

dir = '/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/'
if not os.path.isdir(dir+'training/planes'):
    os.mkdir(dir+'training/planes')
if not os.path.isdir(dir+'testing/planes'):
    os.mkdir(dir+'testing/planes')
    
for file in os.listdir(dir+'training/label_2/'):
    # print(file)
    with open(dir+'training/planes/'+file, 'w') as the_file:
        the_file.write('# Matrix\nWIDTH 4\nHEIGHT 1\n0.00 -1.00 0.00 0.985 ')
    with open(dir+'testing/planes/'+file, 'w') as the_file:
        the_file.write('# Matrix\nWIDTH 4\nHEIGHT 1\n0.00 -1.00 0.00 0.985 ')
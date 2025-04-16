import os

classes = {}

for filename in os.listdir("/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/training/label_2"):
    file = open("/home/luke/NREC/obj_detection/VoxelNeXt/data/custom_data_v4/training/label_2/"+filename, "r")
    lines = file.readlines()
    if len(lines) == 0:
        print(filename)
    for line in lines:
        obj_name = line.split(',')[0]
        if obj_name not in classes:
            classes[obj_name] = 1
        else:
            classes[obj_name] += 1

classes_sorted = sorted(classes.items(), key=lambda x:x[1])
print(classes_sorted)
descending_classes = {}

for class_, num in classes_sorted[::-1]:
    descending_classes[class_] = num
str_ = ''
str_5 = ''
str_15 = ''  
str_cl = ''
str_idx = ''
for idx, class_ in enumerate(list(descending_classes.keys())):
    if descending_classes[class_] >= 10:
        str_+="\"{}\", ".format(str(class_))
        str_5+="\"{}:5\", ".format(str(class_))
        str_15+="\"{}:15\", ".format(str(class_))
        str_cl+="\"{}\":{}, ".format(str(class_), str(idx))
        str_idx+="{}:\"{}\", ".format(str(idx), str(class_))
    
print(str_, '\n')
print(str_5, '\n')
print(str_15, '\n')
print(str_cl, '\n')
print(str_idx, '\n')

print(descending_classes)
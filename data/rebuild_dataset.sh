#!/bin/bash
rm -r custom_data_v4
mkdir -p custom_data_v4/ImageSets
python create_dataset_from_multiple_datasets.py
python generate_planes.py
python create_imagesets.py
cd ..
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
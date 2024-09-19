import argparse
import glob
from pathlib import Path
import os

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch
import open3d as o3d

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.datasets.kitti.kitti_dataset import KittiDataset

class InferDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
    
    # def __len__(self):
    #     return 1
    
    # def __getitem__(self, index):
    #     raise NotImplementedError
    
    def build_frame_data_dict(self, points):
        input_dict = {
            'points': points,
            'frame_id': 0,
            # 'batch_size': 1,
        }
        return self.prepare_data(data_dict=input_dict)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    inference_dataset = InferDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        logger=logger
    )
    
    inference_pc_path = '/home/luke/NREC/obj_detection/VoxelNeXt/data/inference/velodyne'
    assert os.path.exists(inference_pc_path)
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inference_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
        
    with torch.no_grad():
        for file_ in os.listdir(inference_pc_path):
            pcd = o3d.io.read_point_cloud(inference_pc_path + '/' + file_)
            points = np.asarray(pcd.points)
            points = np.hstack([points, np.zeros([points.shape[0], 1])])
            
            input_data_dict = {
                'points': np.asarray(points),
                'frame_id': 0,
            }
            input_data_dict = inference_dataset.prepare_data(data_dict=input_data_dict)
            input_data_dict = inference_dataset.collate_batch([input_data_dict])
            
            load_data_to_gpu(input_data_dict)
            pred_dicts, _ = model.forward(input_data_dict)

            V.draw_scenes(
                points=points, 
                ref_boxes=pred_dicts[0]['pred_boxes'][:1],
                ref_scores=pred_dicts[0]['pred_scores'][:1], 
                # ref_labels=pred_dicts[0]['pred_labels'][:1]
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Inference done.')


if __name__ == '__main__':
    main()

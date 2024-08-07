import argparse
import glob
from pathlib import Path

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

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.datasets.kitti.kitti_dataset import KittiDataset

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index], allow_pickle=True)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        # data_dict = input_dict
        return data_dict
    
    def build_frame_data_dict(self, points):
        input_dict = {
            'points': points,
            'frame_id': 0
        }
        return self.prepare_data(data_dict=input_dict)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    
    # demo_dataset = DemoDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), ext=args.ext, logger=logger
    # )

    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, 
        class_names=cfg.CLASS_NAMES, 
        training=False,
        root_path=Path(args.data_path), 
        logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            
            obj_datadict_list = []
            obj_pred_dict_list = []
            for i in range(len(data_dict['points'])): # number of different objects
                temp_dict = {'gt_names': data_dict['gt_names'][i:i+1], 'gt_boxes': data_dict['gt_boxes'][i:i+1], 'points': data_dict['points'][i]}
                temp_dict['calib'] = data_dict['calib']
                temp_dict['frame_id'] = data_dict['frame_id']
                temp_dict['image_shape'] = data_dict['image_shape']
                temp_dict = demo_dataset.prepare_data(data_dict=temp_dict)
                obj_datadict_list.append(temp_dict)
            
            logger.info(f'Visualized sample index: \t{idx + 1}')
            for obj_data_dict in obj_datadict_list:
                obj_data_dict = demo_dataset.collate_batch([obj_data_dict])
                load_data_to_gpu(obj_data_dict)
                pred_dicts, _ = model.forward(obj_data_dict)
                obj_pred_dict_list.append(pred_dicts[0])
            
            # combine predicted and original data
            combined_data_dict = {'gt_boxes': [], 'points': []}
            for i in range(len(obj_datadict_list)):
                # combined_data_dict['gt_names'].append(obj_datadict_list[i]['gt_names'][0])
                combined_data_dict['gt_boxes'].append(obj_datadict_list[i]['gt_boxes'][0])
                combined_data_dict['points'].append(obj_datadict_list[i]['points'])
            
            combined_data_dict['gt_boxes'] = np.array(combined_data_dict['gt_boxes'])
            combined_data_dict['calib'] = obj_datadict_list[0]['calib']
            combined_data_dict['frame_id'] = obj_datadict_list[0]['frame_id']
            combined_data_dict['image_shape'] = obj_datadict_list[0]['image_shape']
            
            combined_data_dict['points'] = np.concatenate(combined_data_dict['points'], axis=0)
            # combined_data_dict = demo_dataset.prepare_data(data_dict=combined_data_dict)
            
            combined_pred_dict = {'pred_boxes': [], 'pred_scores': [], 'pred_labels': []}
            for i in range(len(obj_pred_dict_list)):
                combined_pred_dict['pred_boxes'].append(obj_pred_dict_list[i]['pred_boxes'][:1])
                combined_pred_dict['pred_scores'].append(obj_pred_dict_list[i]['pred_scores'][:1])
                combined_pred_dict['pred_labels'].append(obj_pred_dict_list[i]['pred_labels'][:1])
            combined_pred_dict['pred_boxes'] = torch.concatenate(combined_pred_dict['pred_boxes'], axis=0)
            
            V.draw_scenes(
                points=combined_data_dict['points'][:, :3], 
                gt_boxes=combined_data_dict['gt_boxes'], 
                ref_boxes=combined_pred_dict['pred_boxes'],
                ref_scores=combined_pred_dict['pred_scores'], 
                # ref_labels=combined_pred_dict['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
                
        # points = np.load(example_file_path, allow_pickle=True)
        # frame_data_dict = dummy_demo_dataset.build_frame_data_dict(points)
        # frame_data_dict = dummy_demo_dataset.collate_batch([frame_data_dict])
        # load_data_to_gpu(frame_data_dict)
        # pred_dicts, _ = model.forward(frame_data_dict)

        # V.draw_scenes(
        #     points=frame_data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
        #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        # )

        # if not OPEN3D_FLAG:
        #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

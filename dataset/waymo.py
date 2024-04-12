import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from utils import read_pickle, read_points, bbox_camera2lidar
from dataset import point_range_filter, data_augment
from torchvision import transforms
from PIL import Image


class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Waymo(Dataset):

    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }

    def __init__(self, data_root, split, pts_prefix='velodyne_reduced', painted=False, cam_sync=False, inference=False):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        if painted or cam_sync:
            info_file = f'painted_waymo_infos_{split}.pkl'
        else:
            info_file = f'waymo_infos_{split}.pkl'
        self.data_infos = read_pickle(os.path.join(data_root, info_file))
        self.sorted_ids = range(len(self.data_infos))
        self.painted = painted
        self.cam_sync = cam_sync
        self.inference = inference
        self.data_aug_config=dict(
            db_sampler=None,
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ), 
            point_range_filter=[-74.88, -74.88, -2, 74.88, 74.88, 4], 
            object_range_filter=[-74.88, -74.88, -2, 74.88, 74.88, 4]
        )

    def remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
        return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info = \
            data_info['image'], data_info['calib'], data_info['annos']
    
        # point cloud input
        velodyne_path = data_info['point_cloud']['velodyne_path']
        pts_path = os.path.join(self.data_root, velodyne_path)
        if self.cam_sync:
            annos_info = data_info['cam_sync_annos']
            if self.painted and not self.inference:
                pts = read_points(pts_path, 11)
            else:
                pts = read_points(pts_path, 11)
                pts = pts[:,:5]
        else:
            pts = read_points(pts_path, 6)
            pts = pts[:,:5]
        
        # calib input: for bbox coordinates transformation between Camera and Lidar.
        # because
        tr_velo_to_cam = calib_info['Tr_velo_to_cam_0'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        # annotations input
        annos_info = self.remove_dont_care(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels), 
            'gt_names': annos_name,
            'difficulty': annos_info['difficulty'],
            'image_info': image_info,
            'calib_info': calib_info
        }
        if self.split in ['train', 'trainval']:
            data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)
        else:
            data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])
        if self.inference:
            images = []
            for i in range(5):
                image = self.get_image(image_info['image_idx'], 'image_' + str(i) + '/')
                images.append(image)
            data_dict['image_info']['images'] = images

        return data_dict

    def get_image(self, idx, camera):
        filename = os.path.join(self.data_root, 'training', camera + ('%s.jpg' % idx))
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch

    def __len__(self):
        return len(self.data_infos)
 

if __name__ == '__main__':
    
    waymo_data = Waymo(data_root='/mnt/ssd1/lifa_rdata/det/kitti', 
                       split='train')
    waymo_data.__getitem__(9)

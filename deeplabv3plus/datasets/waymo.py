import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Waymo(data.Dataset):

    # Based on https://github.com/mcordts/cityscapesScripts
    WaymoClass = namedtuple('WaymoClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        WaymoClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        WaymoClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        WaymoClass('car',                  2, 13, 'void', 0, False, True, (102, 102, 156)),
        WaymoClass('truck',                3, 14, 'void', 0, False, True, (180, 165, 180)),
        WaymoClass('bus',                  4, 15, 'void', 0, False, True, (150, 120, 90)),
        WaymoClass('large_vehicle',        5, 14, 'void', 0, False, True, (111, 74, 0)),
        WaymoClass('bicycle',              6, 18, 'void', 0, False, True, (81, 0, 81)),
        WaymoClass('motorcycle',           7, 17, 'flat', 1, False, False, (128, 64, 128)),
        WaymoClass('trailer',              8, 255, 'flat', 1, False, False, (244, 35, 232)),
        WaymoClass('pedestrian',           9, 11, 'flat', 1, False, True, (250, 170, 160)),
        WaymoClass('cyclist',              10, 12, 'flat', 1, False, True, (230, 150, 140)),
        WaymoClass('motorcyclist',         11, 12, 'construction', 2, False, False, (70, 70, 70)),
        WaymoClass('bird',                 12, 255, 'construction', 2, False, False, (0, 0, 0)),
        WaymoClass('ground_animal',        13, 16, 'construction', 2, False, False, (190, 153, 153)),
        WaymoClass('construction_cone',    14, 255, 'construction', 2, False, True, (0, 0, 0)),
        WaymoClass('pole',                 15, 5, 'construction', 2, False, True, (0, 0, 0)),
        WaymoClass('pedestrian_object',    16, 255, 'construction', 2, False, True, (150, 120, 90)),
        WaymoClass('sign',                 17, 7, 'object', 3, False, False, (153, 153, 153)),
        WaymoClass('traffic_light',        18, 6, 'object', 3, False, True, (153, 153, 153)),
        WaymoClass('building',             19, 2, 'object', 3, False, False, (250, 170, 30)),
        WaymoClass('road',                 20, 0, 'object', 3, False, False, (220, 220, 0)),
        WaymoClass('lane_marker',          21, 0, 'nature', 4, False, False, (107, 142, 35)),
        WaymoClass('road_marker',          22, 0, 'nature', 4, False, False, (152, 251, 152)),
        WaymoClass('sidewalk',             23, 1, 'sky', 5, False, False, (70, 130, 180)),
        WaymoClass('vegetation',           24, 8, 'human', 6, True, False, (220, 20, 60)),
        WaymoClass('sky',                  25, 10, 'human', 6, True, False, (255, 0, 0)),
        WaymoClass('ground',               26, 9, 'vehicle', 7, True, False, (0, 0, 142)),
        WaymoClass('dynamic',              27, 4, 'vehicle', 7, True, False, (0, 0, 70)),
        WaymoClass('static',               28, 3, 'vehicle', 7, True, False, (0, 60, 100)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='training', transform=None):
        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(self.root, 'kitti_format', 'training')

        self.targets_dir = os.path.join(self.root, 'waymo_format', split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        cameras = range(5)
        for cam in cameras:
            img_dir = os.path.join(self.images_dir, 'image_' + str(cam))
            target_dir = os.path.join(self.targets_dir, 'cam_' + str(cam))

            for file_name in os.listdir(target_dir):
                if file_name.endswith('.png'):
                    self.targets.append(os.path.join(target_dir, file_name))
                    img_name = '{}.{}'.format(file_name.split('.')[0], 'jpg')
                    self.images.append(os.path.join(img_dir, img_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)


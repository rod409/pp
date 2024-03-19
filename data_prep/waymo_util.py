# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import multiprocessing
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage import io

def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2',
                   file_tail='.png',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_plane_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='planes',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False,
                      painted=False):
    lidar_folder = 'velodyne'
    ext = '.bin'
    if painted:
        lidar_folder = 'painted_lidar'
        ext = '.npy'
    return get_kitti_info_path(idx, prefix, lidar_folder, ext, training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_pose_path(idx,
                  prefix,
                  training=True,
                  relative_path=True,
                  exist_check=True,
                  use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'pose', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_timestamp_path(idx,
                       prefix,
                       training=True,
                       relative_path=True,
                       exist_check=True,
                       use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'timestamp', '.txt', training,
                               relative_path, exist_check, use_prefix_id)

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def add_difficulty_to_annos(annos):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=bool)
    moderate_mask = np.ones((len(dims), ), dtype=bool)
    hard_mask = np.ones((len(dims), ), dtype=bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff

class WaymoInfoGatherer:
    """
    Parallel version of waymo dataset information gathering.
    Waymo annotation format version like KITTI:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 6
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam0: ...
            P0: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """

    def __init__(self,
                 path,
                 training=True,
                 label_info=True,
                 velodyne=False,
                 calib=False,
                 pose=False,
                 extend_matrix=True,
                 num_worker=8,
                 relative_path=True,
                 with_imageshape=True,
                 painted = False,
                 max_sweeps=5) -> None:
        self.path = path
        self.training = training
        self.label_info = label_info
        self.velodyne = velodyne
        self.calib = calib
        self.pose = pose
        self.extend_matrix = extend_matrix
        self.num_worker = num_worker
        self.relative_path = relative_path
        self.with_imageshape = with_imageshape
        self.max_sweeps = max_sweeps
        self.painted = painted

    def gather_single(self, idx):
        root_path = Path(self.path)
        info = {}
        if self.painted:
            pc_info = {'num_features': 9}
        else:
            pc_info = {'num_features': 6}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if self.velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx,
                self.path,
                self.training,
                self.relative_path,
                use_prefix_id=True,
                painted = self.painted)
        with open(
                get_timestamp_path(
                    idx,
                    self.path,
                    self.training,
                    relative_path=False,
                    use_prefix_id=True)) as f:
            info['timestamp'] = np.int64(f.read())
        image_info['image_path'] = get_image_path(
            idx,
            self.path,
            self.training,
            self.relative_path,
            info_type='image_0',
            file_tail='.jpg',
            use_prefix_id=True)
        if self.with_imageshape:
            img_path = image_info['image_path']
            if self.relative_path:
                img_path = str(root_path / img_path)
            # io using PIL is significantly faster than skimage
            w, h = Image.open(img_path).size
            image_info['image_shape'] = np.array((h, w), dtype=np.int32)
        if self.label_info:
            label_path = get_label_path(
                idx,
                self.path,
                self.training,
                self.relative_path,
                info_type='label_all',
                use_prefix_id=True)
            cam_sync_label_path = get_label_path(
                idx,
                self.path,
                self.training,
                self.relative_path,
                info_type='cam_sync_label_all',
                use_prefix_id=True)
            if self.relative_path:
                label_path = str(root_path / label_path)
                cam_sync_label_path = str(root_path / cam_sync_label_path)
            annotations = get_label_anno(label_path)
            cam_sync_annotations = get_label_anno(cam_sync_label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if self.calib:
            calib_path = get_calib_path(
                idx,
                self.path,
                self.training,
                relative_path=False,
                use_prefix_id=True)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            if self.extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if self.extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            # TODO: naming Tr_velo_to_cam or Tr_velo_to_cam0
            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam1 = np.array([
                float(info) for info in lines[7].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam2 = np.array([
                float(info) for info in lines[8].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam3 = np.array([
                float(info) for info in lines[9].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam4 = np.array([
                float(info) for info in lines[10].split(' ')[1:13]
            ]).reshape([3, 4])
            if self.extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_velo_to_cam1 = _extend_matrix(Tr_velo_to_cam1)
                Tr_velo_to_cam2 = _extend_matrix(Tr_velo_to_cam2)
                Tr_velo_to_cam3 = _extend_matrix(Tr_velo_to_cam3)
                Tr_velo_to_cam4 = _extend_matrix(Tr_velo_to_cam4)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_velo_to_cam1'] = Tr_velo_to_cam1
            calib_info['Tr_velo_to_cam2'] = Tr_velo_to_cam2
            calib_info['Tr_velo_to_cam3'] = Tr_velo_to_cam3
            calib_info['Tr_velo_to_cam4'] = Tr_velo_to_cam4
            info['calib'] = calib_info

        if self.pose:
            pose_path = get_pose_path(
                idx,
                self.path,
                self.training,
                relative_path=False,
                use_prefix_id=True)
            info['pose'] = np.loadtxt(pose_path)

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['camera_id'] = info['annos'].pop('score')
            add_difficulty_to_annos(info['annos'])
            info['cam_sync_annos'] = cam_sync_annotations
            add_difficulty_to_annos(info['cam_sync_annos'])
            # NOTE: the 2D labels do not have strict correspondence with
            # the projected 2D lidar labels
            # e.g.: the projected 2D labels can be in camera 2
            # while the most_visible_camera can have id 4
            info['cam_sync_annos']['camera_id'] = info['cam_sync_annos'].pop(
                'score')

        sweeps = []
        prev_idx = idx
        while len(sweeps) < self.max_sweeps:
            prev_info = {}
            prev_idx -= 1
            prev_info['velodyne_path'] = get_velodyne_path(
                prev_idx,
                self.path,
                self.training,
                self.relative_path,
                exist_check=False,
                use_prefix_id=True,
                painted = self.painted)
            if_prev_exists = osp.exists(
                Path(self.path) / prev_info['velodyne_path'])
            if if_prev_exists:
                with open(
                        get_timestamp_path(
                            prev_idx,
                            self.path,
                            self.training,
                            relative_path=False,
                            use_prefix_id=True)) as f:
                    prev_info['timestamp'] = np.int64(f.read())
                prev_info['image_path'] = get_image_path(
                    prev_idx,
                    self.path,
                    self.training,
                    self.relative_path,
                    info_type='image_0',
                    file_tail='.jpg',
                    use_prefix_id=True)
                prev_pose_path = get_pose_path(
                    prev_idx,
                    self.path,
                    self.training,
                    relative_path=False,
                    use_prefix_id=True)
                prev_info['pose'] = np.loadtxt(prev_pose_path)
                sweeps.append(prev_info)
            else:
                break
        info['sweeps'] = sweeps

        return info

    def gather(self, image_ids):
        if not isinstance(image_ids, list):
            image_ids = list(range(image_ids))
        #with multiprocessing.Pool(self.num_worker) as p:
        #    image_infos = tqdm(p.imap(self.gather_single, image_ids), total=self.num_worker)
        image_infos = tqdm(map(self.gather_single, image_ids))
        return list(image_infos)

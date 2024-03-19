import argparse
import os
import functools
import waymo_converter as waymo
import waymo_util
from pathlib import Path
import numpy as np
import pickle
import multiprocessing
#from update_pkl_infos import update_pkl_infos
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pathlib
import box_np_ops


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5,
                    painted=False):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 5. Here we store pose information of these frames
            for later use.
    """

    splits = [
        'training', 'validation', 'testing', 'testing_3d_camera_only_detection'
    ]
    for i, split in enumerate(splits):
        load_dir = os.path.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = os.path.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = os.path.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split
                       in ['testing', 'testing_3d_camera_only_detection']))
        converter.convert()

    waymo.create_ImageSets_img_ids(os.path.join(out_dir, 'kitti_format'), splits)
    # Generate waymo infos
    out_dir = os.path.join(out_dir, 'kitti_format')
    create_waymo_info_file(
        out_dir, info_prefix, max_sweeps=max_sweeps, workers=workers, painted=painted)
    info_train_path = os.path.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = os.path.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = os.path.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = os.path.join(out_dir, f'{info_prefix}_infos_test.pkl')
    #update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_train_path)
    #update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_val_path)
    #update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_trainval_path)
    #update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_test_path)
    '''GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()'''
    #create_groundtruth_database(out_dir, painted=painted)

def create_waymo_info_file(data_path,
                           pkl_prefix='waymo',
                           save_path=None,
                           relative_path=True,
                           max_sweeps=5,
                           workers=8,
                           painted=False):
    """Create info file of waymo dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'waymo'.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        max_sweeps (int, optional): Max sweeps before the detection frame
            to be used. Default: 5.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    waymo_infos_gatherer_trainval = waymo_util.WaymoInfoGatherer(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        pose=True,
        relative_path=relative_path,
        max_sweeps=max_sweeps,
        num_worker=workers,
        painted=painted)
    waymo_infos_gatherer_test = waymo_util.WaymoInfoGatherer(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        pose=True,
        relative_path=relative_path,
        max_sweeps=max_sweeps,
        num_worker=workers,
        painted=painted)
    '''num_points_in_gt_calculater = _NumPointsInGTCalculater(
        data_path,
        relative_path,
        num_features=6,
        remove_outside=False,
        num_worker=workers)'''

    waymo_infos_train = waymo_infos_gatherer_trainval.gather(train_img_ids)
    #num_points_in_gt_calculater.calculate(waymo_infos_train)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Waymo info train file is saved to {filename}')
    with open(filename, 'wb') as pickle_file:
        pickle.dump(waymo_infos_train, pickle_file)
    waymo_infos_val = waymo_infos_gatherer_trainval.gather(val_img_ids)
    #num_points_in_gt_calculater.calculate(waymo_infos_val)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Waymo info val file is saved to {filename}')
    with open(filename, 'wb') as pickle_file:
        pickle.dump(waymo_infos_val, pickle_file)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Waymo info trainval file is saved to {filename}')
    with open(filename, 'wb') as pickle_file:
        pickle.dump(waymo_infos_train + waymo_infos_val, pickle_file)
    waymo_infos_test = waymo_infos_gatherer_test.gather(test_img_ids)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Waymo info test file is saved to {filename}')
    with open(filename, 'wb') as pickle_file:
        pickle.dump(waymo_infos_test, pickle_file)

def create_single(info,
                  root_path,
                  used_classes=None,
                  database_save_path=None,
                  relative_path=True,
                  lidar_only=False,
                  bev_only=False,
                  coors_range=None):
        db_info = {}
        velodyne_path = info['point_cloud']['velodyne_path']
        if relative_path:
            # velodyne_path = str(root_path / velodyne_path) + "_reduced"
            velodyne_path = str(root_path / velodyne_path)
        num_features = 6
        if 'pointcloud_num_features' in info['point_cloud']:
            num_features = info['point_cloud']['pointcloud_num_features']
        points = np.fromfile(
            velodyne_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        image_idx = info['image']["image_idx"]
        rect = info['calib']['R0_rect']
        P0 = info['calib']['P0']
        Trv2c = info['calib']['Tr_velo_to_cam']
        if not lidar_only:
            points = box_np_ops.remove_outside_points(points, rect, Trv2c, P0,
                                                        info['image']["image_shape"])

        annos = info["annos"]
        names = annos["name"]
        bboxes = annos["bbox"]
        difficulty = annos["difficulty"]
        gt_idxes = annos["index"]
        num_obj = np.sum(annos["index"] >= 0)
        rbbox_cam = anno_to_rbboxes(annos)[:num_obj]
        rbbox_lidar = box_np_ops.box_camera_to_lidar(rbbox_cam, rect, Trv2c)
        if bev_only: # set z and h to limits
            assert coors_range is not None
            rbbox_lidar[:, 2] = coors_range[2]
            rbbox_lidar[:, 5] = coors_range[5] - coors_range[2]
        
        group_dict = {}
        group_ids = np.full([bboxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(bboxes.shape[0], dtype=np.int64)
        point_indices = box_np_ops.points_in_rbbox(points, rbbox_lidar)
        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{gt_idxes[i]}.bin"
            filepath = database_save_path / filename
            gt_points = points[point_indices[:, i]]

            gt_points[:, :3] -= rbbox_lidar[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)
            if names[i] in used_classes:
                if relative_path:
                    db_path = str(database_save_path.stem + "/" + filename)
                else:
                    db_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": gt_idxes[i],
                    "box3d_lidar": rbbox_lidar[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }

                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                #if local_group_id not in group_dict:
                #    group_dict[local_group_id] = group_counter
                #    group_counter += 1
                #db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
        return db_info

def create_groundtruth_database(data_path,
                                info_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                lidar_only=False,
                                bev_only=False,
                                workers=8,
                                coors_range=None,
                                painted=False):
    root_path = pathlib.Path(data_path)
    if info_path is None:
        info_path = root_path / 'waymo_infos_train.pkl'
    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = pathlib.Path(database_save_path)
    if db_info_save_path is None:
        db_info_save_path = root_path / "waymo_dbinfos_train.pkl"
    database_save_path.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'rb') as f:
        waymo_infos = pickle.load(f)
    all_db_infos = {}
    if used_classes is None:
        used_classes = list(waymo.get_classes())
        #used_classes.pop(used_classes.index('DontCare'))
    for name in used_classes:
        all_db_infos[name] = []
    #group_counter = 0
    single = functools.partial(create_single, root_path=root_path,
                  used_classes=used_classes,
                  database_save_path=database_save_path,
                  relative_path=relative_path,
                  lidar_only=lidar_only,
                  bev_only=bev_only,
                  coors_range=coors_range)
    results = process_map(single, waymo_infos, max_workers=workers)
    for db_info in results:
        if 'name' in db_info: 
            all_db_infos[db_info['name']].append(db_info)
            
        
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

def anno_to_rbboxes(anno):
    loc = anno["location"]
    dims = anno["dimensions"]
    rots = anno["rotation_y"]
    rbboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    return rbboxes

def main(args):
    prefix = 'waymo'
    if args.painted:
        prefix = 'painted_waymo'
    waymo_data_prep(args.waymo_root, prefix, '1.0', args.waymo_root, args.workers, painted=args.painted)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--waymo_root', help='your data root for the waymo dataset', required=True)
    parser.add_argument('--workers', default=4, help='number of processes')
    parser.add_argument('--painted', action='store_true', help='if using painted lidar points')
    args = parser.parse_args()
    main(args)


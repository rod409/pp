import argparse
import os
import waymo_util
from pathlib import Path
import pickle


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

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

    waymo_infos_train = waymo_infos_gatherer_trainval.gather(train_img_ids)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Waymo info train file is saved to {filename}')
    with open(filename, 'wb') as pickle_file:
        pickle.dump(waymo_infos_train, pickle_file)
    waymo_infos_val = waymo_infos_gatherer_trainval.gather(val_img_ids)
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


def main(args):
    prefix = 'waymo'
    if args.painted:
        prefix = 'painted_waymo'
    out_dir = os.path.join(args.waymo_root, 'kitti_format')
    create_waymo_info_file(out_dir, prefix, workers=args.workers, painted=args.painted)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--waymo_root', help='your data root for the waymo dataset', required=True)
    parser.add_argument('--workers', default=4, help='number of processes')
    parser.add_argument('--painted', action='store_true', help='if using painted lidar points')
    args = parser.parse_args()
    main(args)


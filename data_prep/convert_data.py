import os
import waymo_converter as waymo
import argparse

def convert_data(root_path,
                    out_dir,
                    workers):
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

def main(args):
    convert_data(args.waymo_root, args.waymo_root, args.workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--waymo_root', help='your data root for the waymo dataset', required=True)
    parser.add_argument('--workers', default=4, help='number of processes')
    args = parser.parse_args()
    main(args)
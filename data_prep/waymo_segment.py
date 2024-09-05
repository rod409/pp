import os
import argparse
import PIL
import numpy as np
import tensorflow as tf
import pathlib
from glob import glob
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import numpy as np
import dask.dataframe as dd
from tqdm.contrib.concurrent import process_map
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
#from waymo_open_dataset import v2
from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
#from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics
from waymo_open_dataset.utils import camera_segmentation_utils

class WaymoSegment(object):
    def __init__(self, load_dir, save_dir, workers=64, prefix=0):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.tfrecord_pathnames = sorted(glob(os.path.join(self.load_dir, '*.tfrecord')))
        self.workers = workers
        self.prefix = prefix
        for i in range(5):
            pathlib.Path(os.path.join(save_dir, 'cam_'+str(i))).mkdir(parents=True, exist_ok=True)


    def save_semantic_label(self, file_idx, frame_idx, label, camera_id):
        label_path = f'{self.save_dir}/cam_{str(camera_id)}/' + \
            f'{self.prefix}{str(file_idx).zfill(3)}' + \
            f'{str(frame_idx).zfill(3)}.npy'
        np.save(label_path, label)

    def convert_one(self, file_idx):
        dataset = tf.data.TFRecordDataset(self.tfrecord_pathnames[file_idx], compression_type='')
        frames_with_seg = []
        sequence_id = None
        total = 0
        labeled = 0
        frame_ids = []
        camera_order = [open_dataset.CameraName.FRONT,
                        open_dataset.CameraName.FRONT_LEFT,
                        open_dataset.CameraName.FRONT_RIGHT,
                        open_dataset.CameraName.SIDE_LEFT,
                        open_dataset.CameraName.SIDE_RIGHT]
        for frame_idx, data in enumerate(dataset):
            total += 1
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            # Save frames which contain CameraSegmentationLabel messages. We assume that
            # if the first image has segmentation labels, all images in this frame will.
            if frame.images[0].camera_segmentation_label.panoptic_label:
                labeled += 1
                frames_with_seg.append(frame)
                frame_ids.append(frame_idx)
                if sequence_id is None:
                    sequence_id = frame.images[0].camera_segmentation_label.sequence_id
                segmentation_proto_dict = {image.name : image.camera_segmentation_label for image in frame.images}
                segmentation_protos_ordered = []
                segmentation_protos_ordered.append([segmentation_proto_dict[name] for name in camera_order])
                for i in range(len(segmentation_protos_ordered)):
                    for camera in camera_order:
                        cam_idx = camera-1
                        panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
                            segmentation_protos_ordered[i][cam_idx]
                        )
                        semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                            panoptic_label,
                            segmentation_protos_ordered[i][cam_idx].panoptic_label_divisor
                        )

                        #np.save(f'{str(i).zfill(3)}{str(cam_idx)}.npy', semantic_label)
                        self.save_semantic_label(file_idx, frame_idx, semantic_label, cam_idx)

    def convert(self):
        process_map(self.convert_one, range(len(self.tfrecord_pathnames)), max_workers=self.workers)

def main(args):
    train_root = os.path.join(args.waymo_root, 'waymo_format', 'training')
    val_root = os.path.join(args.waymo_root, 'waymo_format', 'validation')
    waymo = WaymoSegment(train_root, train_root, workers=args.workers, prefix=0)
    waymo.convert()
    waymo = WaymoSegment(val_root, val_root, workers=args.workers, prefix=1)
    waymo.convert()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--waymo_root', help='your data root for the waymo dataset', required=True)
    parser.add_argument('--workers', default=4, help='number of processes', type=int)
    args = parser.parse_args()
    main(args)


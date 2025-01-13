import argparse
import numpy as np
import os
import torch
import copy
from collections import namedtuple
from tqdm import tqdm
import time
import csv
import onnx
import onnxruntime as ort
from dataset import Waymo, get_dataloader
from model import PointPillarsPre, PointPillarsPos
import deeplabv3plus.network as network
from painting.painting import Painter
from utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d_camera, iou_bev
from evaluate import do_eval

def convert_calib(calib, cuda):
    result = {}
    result['R0_rect'] = torch.from_numpy(calib['R0_rect'])
    for i in range(5):
        result['P' + str(i)] = torch.from_numpy(calib['P' + str(i)])
        result['Tr_velo_to_cam_' + str(i)] = torch.from_numpy(calib['Tr_velo_to_cam_' + str(i)])
    return change_calib_device(result, cuda)

def change_calib_device(calib, cuda):
    result = {}
    if cuda:
        device = 'cuda'
    else:
        device='cpu'
    result['R0_rect'] = calib['R0_rect'].to(device=device, dtype=torch.float)
    for i in range(5):
        result['P' + str(i)] = calib['P' + str(i)].to(device=device, dtype=torch.float)
        result['Tr_velo_to_cam_' + str(i)] = calib['Tr_velo_to_cam_' + str(i)].to(device=device, dtype=torch.float)
    return result

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main(args):
    val_dataset = Waymo(data_root=args.data_root,
                        split='val', painted=args.painted, cam_sync=args.cam_sync, inference=True)
    val_dataloader, _ = get_dataloader(dataset=val_dataset, 
                                    batch_size=1, 
                                    num_workers=args.num_workers,
                                    rank=0,
                                    world_size=1,
                                    shuffle=False)
    CLASSES = Waymo.CLASSES
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}

    ort_sess = ort.InferenceSession(args.lidar_detector)
    input_pillars_name = ort_sess.get_inputs()[0].name
    input_coors_batch_name = ort_sess.get_inputs()[1].name
    input_npoints_per_pillar_name = ort_sess.get_inputs()[2].name
    output_name = ort_sess.get_inputs()[0].name
    if not args.no_cuda:
        model_pre = PointPillarsPre().cuda()
        model_post = PointPillarsPos(nclasses=len(CLASSES)).cuda()
    else:
        model_pre = PointPillarsPre()
        model_post = PointPillarsPos(nclasses=len(CLASSES))
    model_pre.eval()
    model_post.eval()
    PaintArgs = namedtuple('PaintArgs', ['training_path', 'model_path', 'cam_sync'])
    painting_args = PaintArgs(os.path.join(args.data_root, 'training'), args.segmentor, args.cam_sync)
    painter = Painter(painting_args, onnx=True)
    deeplab = painter.model
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    saved_submit_path = os.path.join(saved_path, 'submit')
    os.makedirs(saved_submit_path, exist_ok=True)

    pcd_limit_range = torch.tensor([-74.88, -74.88, -2, 74.88, 74.88, 4])

    with torch.inference_mode():
        format_results = {}
        print('Predicting and Formatting the results.')
        latency_results = []
        for i, data_dict in enumerate(tqdm(val_dataloader)):
            data_dict['batched_calib_info'][0] = convert_calib(data_dict['batched_calib_info'][0], not args.no_cuda)
            if not args.no_cuda:
                # move the tensors to the cuda
                data_dict['batched_pts'][0].to(device='cuda')
                for i in range(len(data_dict['batched_images'][0])):
                    data_dict['batched_images'][0][i] = data_dict['batched_images'][0][i].to(device='cuda')
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
                    
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            #batched_images = data_dict['batched_images'][0]
            scores_from_cam = []
            start_time = time.perf_counter()
            for i in range(len(data_dict['batched_images'][0])):
                input_image_name = deeplab.get_inputs()[0].name
                input_data = {input_image_name: to_numpy(data_dict['batched_images'][0][i])}
                segmentation_score = deeplab.run(None, input_data)
                segmentation_score = [torch.from_numpy(item) for item in segmentation_score]
                #segmentation_score = deeplab(data_dict['batched_images'][0][i])[0]
                scores_from_cam.append(painter.get_score(segmentation_score[0].squeeze(0)))

            points = painter.augment_lidar_class_scores_both(scores_from_cam, batched_pts[0], data_dict['batched_calib_info'][0])
            pillars, coors_batch, npoints_per_pillar = model_pre(batched_pts=[points])
            input_data = {input_pillars_name: to_numpy(pillars),
                        input_coors_batch_name: to_numpy(coors_batch),
                        input_npoints_per_pillar_name: to_numpy(npoints_per_pillar)}
            result = ort_sess.run(None, input_data)
            result = [torch.from_numpy(item) for item in result]
            batch_results = model_post(result)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            latency_results.append(str(total_time) + '\n')
            for j, result in enumerate(batch_results):
                format_result = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }
                
                calib_info = data_dict['batched_calib_info'][j]
                image_info = data_dict['batched_img_info'][j]
                idx = data_dict['batched_img_info'][j]['image_idx']
                
                calib_info = change_calib_device(calib_info, False)
                result_filter = keep_bbox_from_image_range(result, calib_info, 5, image_info, args.cam_sync)
                #result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
                
                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']
                bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                for lidar_bbox, label, score, bbox2d, camera_bbox in \
                    zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                    format_result['name'].append(LABEL2CLASSES[label.item()])
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                    format_result['alpha'].append(alpha.item())
                    format_result['bbox'].append(bbox2d.tolist())
                    format_result['dimensions'].append(camera_bbox[3:6])
                    format_result['location'].append(camera_bbox[:3])
                    format_result['rotation_y'].append(camera_bbox[6].item())
                    format_result['score'].append(score.item())
                
                #write_label(format_result, os.path.join(saved_submit_path, f'{idx:06d}.txt'))

                if len(format_result['dimensions']) > 0:
                    format_result['dimensions'] = torch.stack(format_result['dimensions'])
                    format_result['location'] = torch.stack(format_result['location'])
                format_results[idx] = {k:np.array(v) for k, v in format_result.items()}
        
        write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))
    
    with open('latency.txt', 'w', newline='') as f:
        f.writelines(latency_results)
    print('Evaluating.. Please wait several seconds.')
    do_eval(format_results, val_dataset.data_infos, CLASSES, saved_path, cam_sync=args.cam_sync)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', help='your data root for waymo')
    parser.add_argument('--lidar_detector', default='pp.onnx', help='your lidar model onnx file')
    parser.add_argument('--segmentor', help='your segmentation model checkpoint', required=True)
    parser.add_argument('--saved_path', default='results', help='your saved path for predicted results')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--painted', action='store_true', help='if using painted lidar points')
    parser.add_argument('--cam_sync', action='store_true', help='only use objects visible to a camera')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
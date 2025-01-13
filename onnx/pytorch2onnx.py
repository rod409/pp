import argparse
import numpy as np
import os
import sys
import torch


CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR))

from model import PointPillarsCore

def main(args):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }

    if not args.no_cuda:
        model = PointPillarsCore(nclasses=len(CLASSES), painted=True).cuda()
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = PointPillarsCore(nclasses=len(CLASSES), painted=True)
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    model.eval()


    print('start to transform pytorch model to onnx')
    max_pillars = 32000
    max_points_per_pillar = 20
    pillars = torch.randn(max_pillars, max_points_per_pillar, 11)
    coors_batch = torch.randint(0, 216, (max_pillars, 4))
    coors_batch[:, 0] = 0
    npoints_per_pillar = torch.randint(0, max_points_per_pillar, (max_pillars, ))
    npoints_per_pillar = npoints_per_pillar.to(torch.int32)
    if not args.no_cuda:
        pillars = pillars.cuda()
        coors_batch = coors_batch.cuda()
        npoints_per_pillar = npoints_per_pillar.cuda()

    torch.onnx.export(model, (pillars, coors_batch, npoints_per_pillar), args.saved_onnx_path, 
                      export_params=True, opset_version=11, do_constant_folding=True, 
                      input_names=['input_pillars', 'input_coors_batch', 'input_npoints_per_pillar'],
                      dynamic_axes={'input_pillars': {0: 'pillar_num'}, 
                                    'input_coors_batch': {0: 'pillar_num'}, 
                                    'input_npoints_per_pillar': {0: 'pillar_num'}},
                      output_names=['output_x'])
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', help='your checkpoint for pointpillars')
    parser.add_argument('--saved_onnx_path', default='../pp.onnx',
                        help='your saved onnx path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
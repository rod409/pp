import argparse
import numpy as np
import os
import sys
import torch


CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR))

from model import PointPillarsCore
import deeplabv3plus.network as network

def main(args):
    model = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=19, output_stride=16)
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint["model_state"])
    input_image = torch.rand(1, 3, 1920, 1280)
    torch.onnx.export(model, input_image, args.saved_onnx_path, 
                      export_params=True, opset_version=11, do_constant_folding=True, 
                      input_names=['input_image'],
                      dynamic_axes={'input_image': [0, 2, 3]},
                      output_names=['output_deep'])
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', help='your checkpoint for pointpillars')
    parser.add_argument('--saved_onnx_path', default='../deeplabv3+.onnx',
                        help='your saved onnx path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
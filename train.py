import argparse
import os
import torch
from tqdm import tqdm
import pdb

from utils import setup_seed
from dataset import Waymo, get_dataloader
from model import PointPillars
from loss import Loss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import math


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)

def main(rank, args, world_size):
    setup_seed()
    train_dataset = Waymo(data_root=args.data_root,
                          split='train', painted=args.painted, cam_sync=args.cam_sync, interval = args.load_interval)
    train_dataloader, sampler = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      rank=rank,
                                      world_size=world_size,
                                      shuffle=True)

    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses, painted=args.painted).cuda()
        pointpillars = DDP(pointpillars, device_ids=[rank], output_device=rank)
    else:
        pointpillars = PointPillars(nclasses=args.nclasses, painted=args.painted)
    loss_func = Loss()

    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  
                                                    milestones=[20, 23],
                                                    gamma=0.1)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)
    
    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        first_epoch = checkpoint["epoch"] + 1
        pointpillars.module.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    for epoch in range(first_epoch, args.max_epoch):
        sampler.set_epoch(epoch)
        if rank == 0:
            print('=' * 20, epoch, '=' * 20)
        train_step = 0
        ave_loss = 0
        with tqdm(total=len(train_dataloader), disable=rank != 0) as pbar:
            for i, data_dict in enumerate(train_dataloader):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                optimizer.zero_grad()

                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)
                
                loss = loss_dict['total_loss']
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
                optimizer.step()

                global_step = epoch * len(train_dataloader) + train_step + 1

                if global_step % args.log_freq == 0 and rank == 0:
                    save_summary(writer, loss_dict, global_step, 'train',
                                lr=optimizer.param_groups[0]['lr'], 
                                momentum=optimizer.param_groups[0]['betas'][0])
                train_step += 1
                pbar.update(1)
                ave_loss += loss.item()
                pbar.set_postfix({
                    'Avg. loss': ave_loss/(i+1),
                    'lr': scheduler.get_last_lr()
                    })
        scheduler.step()
        if (epoch + 1) % args.ckpt_freq_epoch == 0 and rank == 0:
            checkpoint = {"epoch": epoch,
                "model_state_dict": pointpillars.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}
            torch.save(checkpoint, os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--sched_max_epoch', type=int, default=60)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--load_interval', type=int, default=1, help='the training interval for loading items')
    parser.add_argument('--ckpt', default='', help='your model checkpoint')
    parser.add_argument('--ckpt_freq_epoch', type=int, default=5)
    parser.add_argument('--painted', action='store_true', help='if using painted lidar points')
    parser.add_argument('--cam_sync', action='store_true', help='only use objects visible to a camera')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--local-rank', default=0, type=int)
    args = parser.parse_args()
    torch.distributed.init_process_group("nccl", init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    main(rank, args, world_size)

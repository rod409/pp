# Introduction
This repository is an implementation of [PointPainting](https://arxiv.org/abs/1911.10150) on the [Waymo Open Dataset](https://waymo.com/open/).

# Dataset
Download v1.4.1 of the [Waymo Open Dataset](https://waymo.com/open/download/) following the instruction. Format the segments under the following folder structure.
    
    waymo
        |- waymo_format
            |- training
            |- validation
            |- testing


# Environment
Create a conda environment 

```
  conda create -n pp python=3.10
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  conda install tqdm
  conda install numba==0.57
  pip install opencv-python
  pip install open3d
  pip install tensorboard
  pip install waymo-open-dataset-tf-2-11-0==1.6.1
```
## Compile ops
```
    conda activate pp
    cd ops
    python setup.py develop
```
# Preparing the Dataset
First convert the dataset to the KITTI format. This will create a kitti_format folder under your waymo directory.
```
  cd data_prep
  python create_data.py --waymo_root [path/to/waymo] --painted --convert
```
Next paint the lidar points using a trained segmentation model.
```
cd painting
python painting.py --training_path [path/to/waymo]/kitti_format/training/ --model_path [path/to/segmentation/model]
```
Create the info file used for training
```
  cd data_prep
  python create_data.py --waymo_root [path/to/waymo] --painted --create_info
```

# Training
To train on painted lidar points.
```
  conda activate pp
  python -m torch.distributed.launch --nproc_per_node=[gpus] train.py --data_root [path/to/waymo]/kitti_format/  --painted --cam_sync --save-path [checkpoint/path] --max_epoch [num of epochs] --chkpt_freq_epoch [freq]
```
To evaluate the mAP.
```
conda activate pp
python evaluate.py --ckpt [checkpoint/path] --data_root [path/to/waymo]/kitti_format/ --painted --cam_sync
```
# Acknowledements

This repository makes use of the open source from
[PointPillars](https://github.com/zhulf0804/pointpillars), [MMDet3D](https://github.com/open-mmlab/mmdetection3d), [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch), and [PointPainting](https://github.com/Song-Jingyu/PointPainting).

FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install tqdm
RUN pip install numba==0.57
RUN pip install opencv-python
RUN pip install open3d
RUN pip install tensorboard
RUN pip install waymo-open-dataset-tf-2-11-0==1.6.1


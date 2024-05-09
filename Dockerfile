FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install tqdm
RUN pip install numba
RUN pip install opencv-python
RUN pip install open3d
RUN pip install tensorboard
RUN pip install scikit-image
RUN pip install ninja
RUN pip install visdom

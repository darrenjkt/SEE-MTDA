# Base image is from scrin/dev-spconv:latest but one of the apt packages are discontinued
# I entered the container, deleted the line with the pkg from /etc/apt/sources.list 
# and saved it as the following below
FROM darrenjkt/spconv:scrin-dev-ubuntu20.04

# Install ST3D
WORKDIR /
RUN git clone https://github.com/CVMI-Lab/ST3D.git
WORKDIR /ST3D
RUN pip install -r requirements.txt 

# The TORCH_CUDA_ARCH_LIST="6.1 7.5" is specifically for 1080Ti and 2080Ti setup. Though I don't think I need it. Keep it here in case.
RUN python setup.py develop

RUN python3 -m pip install pip==21.0.1
RUN pip3 install --upgrade ipykernel

# Surface completion dependencies
RUN pip3 install --ignore-installed PyYAML easydict
RUN pip3 install pycocotools open3d opencv-python setproctitle pathos

# Nuscenes and waymo dataset
RUN pip3 install nuscenes-devkit==1.0.5 waymo-open-dataset-tf-2-2-0 --user

# Generic screen support
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6

# Clean image
RUN apt-get clean && rm -rf /var/lib/apt/lists/* 

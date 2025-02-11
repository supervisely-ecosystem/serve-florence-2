FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install packaging==22.0

RUN pip3 install transformers==4.46.2
RUN pip3 install timm==1.0.11
RUN pip3 install flash_attn==2.6.3
RUN pip3 install einops==0.8.0
RUN pip3 install accelerate==1.2.1

RUN apt-get update
RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN pip install opencv-python

RUN python3 -m pip install supervisely==6.73.251

RUN pip3 install matplotlib
RUN pip3 install ipywidgets

RUN apt-get -y install curl
RUN apt-get -y install wget

RUN apt -y install wireguard iproute2
RUN apt-get install nano

LABEL "role"="development"
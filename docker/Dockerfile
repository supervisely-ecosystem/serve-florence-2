FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y python3-pip git curl wget ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev

RUN pip3 install --upgrade pip
COPY dev_requirements.txt dev_requirements.txt
RUN pip3 install -r dev_requirements.txt

ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV CUDA_HOME=/usr/local/cuda-12.1

COPY download_weights.py /app/download_weights.py
COPY models.json /app/models.json
WORKDIR /app
RUN python3 download_weights.py

LABEL python_sdk_version="6.73.486"
# FROM nvidia/cuda:01.6.1-cudnn8-runtime-ubuntu20.04
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
   && apt-get -y install --no-install-recommends \ 
   git openssh-server \
   # Clean up
   && apt-get autoremove -y \
   && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog

RUN pip3 install numpy jupyterlab matplotlib \
   pandas transforms3d pyquaternion vit-pytorch
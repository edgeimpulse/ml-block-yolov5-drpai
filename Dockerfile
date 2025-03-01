# syntax = docker/dockerfile:experimental
ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.2-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8
# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Avoid confirmation dialogs
ENV DEBIAN_FRONTEND=noninteractive
# Makes Poetry behave more like npm, with deps installed inside a .venv folder
# See https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# CUDA drivers
SHELL ["/bin/bash", "-c"]
COPY ./dependencies/install_cuda.sh ./install_cuda.sh
RUN ./install_cuda.sh && \
    rm install_cuda.sh

# System dependencies
RUN apt update && apt install -y wget git python3 python3-pip zip protobuf-compiler vim libgl1 libhdf5-serial-dev pkg-config

# Install CMake (required for onnx 1.8.1)
COPY dependencies/install_cmake.sh install_cmake.sh
RUN /bin/bash install_cmake.sh && \
    rm install_cmake.sh

# Pin pip / setuptools
RUN python3 -m pip install "pip==21.3.1" "setuptools==62.6.0"

# YOLOv5 (v5.0-with-freeze-2 branch)
RUN wget https://cdn.edgeimpulse.com/model-repos/yolov5-9a5b7e9f5a28817cb1542e58bd1cf4fbd8504966.zip && \
    unzip -q yolov5-9a5b7e9f5a28817cb1542e58bd1cf4fbd8504966.zip && \
    mv yolov5-training-9a5b7e9f5a28817cb1542e58bd1cf4fbd8504966 yolov5 && \
    rm yolov5-9a5b7e9f5a28817cb1542e58bd1cf4fbd8504966.zip

# Local dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Install TensorFlow
COPY dependencies/install_tensorflow.sh install_tensorflow.sh
RUN /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh

# Install TensorFlow addons
COPY dependencies/install_tensorflow_addons.sh install_tensorflow_addons.sh
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/app/wheels \
    /bin/bash install_tensorflow_addons.sh && \
    rm install_tensorflow_addons.sh

COPY requirements-nvidia.txt ./
RUN pip3 install -r requirements-nvidia.txt

# install onnx-tf (needs to be after tensorflow) and force downgrade protobuf
RUN pip3 install onnx-tf==1.8.0 protobuf==3.20.1

# Grab pretrained weights
RUN wget -O yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt && \
    wget -O yolov5m.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt && \
    wget -O yolov5l.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt

# Download some files that are pulled in, so we can run w/o network access
RUN mkdir -p /root/.config/Ultralytics/ && wget -O /root/.config/Ultralytics/Arial.ttf https://ultralytics.com/assets/Arial.ttf

WORKDIR /scripts

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]

#!/bin/bash

export DATA_ROOT_DIR=/Volumes/LaCie/Dropbox/recent/machine_learning/denoising_autoencoders/2019_spiral
export SRC_ROOT_DIR=${HOME}/Documents/code/denoising_autoencoders/2019_spiral

docker run -it -p 8888:8888 -v ${SRC_ROOT_DIR}:/current_project -v ${SRC_ROOT_DIR}/notebooks:/notebooks -v ${DATA_ROOT_DIR}:/current_project/data  tensorflow/tensorflow

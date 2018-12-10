#!/bin/bash

export SRC_ROOT_DIR=${HOME}/Documents/code/denoising_autoencoder/2019_spiral
export DATA_ROOT_DIR=${SRC_ROOT_DIR}/data

export DOCKER_CONTAINER_NAME=dae:v1

docker run -it -p 8888:8888 -v ${SRC_ROOT_DIR}:/current_project -v ${SRC_ROOT_DIR}/notebooks:/notebooks -v ${DATA_ROOT_DIR}:/current_project/data ${DOCKER_CONTAINER_NAME}

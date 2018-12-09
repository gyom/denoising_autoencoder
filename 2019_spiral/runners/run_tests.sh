#!/bin/bash


export SRC_ROOT_DIR=${HOME}/Documents/code/denoising_autoencoder/2019_spiral
export DATA_ROOT_DIR=${SRC_ROOT_DIR}/data

# export DOCKER_CONTAINER_NAME=tensorflow/tensorflow
# Use this to have testing and progressbar.
export DOCKER_CONTAINER_NAME=dae:v1

docker run -it  -v ${SRC_ROOT_DIR}:/current_project  -v ${DATA_ROOT_DIR}:/current_project/data ${DOCKER_CONTAINER_NAME}  \
    nosetests /current_project/src/denoising_autoencoder/logsumexp.py

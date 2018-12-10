#!/bin/bash


export SRC_ROOT_DIR=${HOME}/Documents/code/denoising_autoencoder/2019_spiral

# If this took a lot of space, we would write there, but it's not going to be
# that big.
#export DATA_ROOT_DIR=/Volumes/LaCie/Dropbox/recent/machine_learning/denoising_autoencoder/2019_spiral
export DATA_ROOT_DIR=${SRC_ROOT_DIR}/data

# Not needed for this task.
# -v /Volumes/LaCie/Dropbox/recent/machine_learning/RL2017_deepmindprep/experiments:/current_project/experiments
# -v /Volumes/LaCie/Dropbox/recent/machine_learning/RL2017_deepmindprep/checkpoints:/current_project/checkpoints

# Use this to have less features.
# export DOCKER_CONTAINER_NAME=tensorflow/tensorflow
# Use this to have testing and progressbar.
export DOCKER_CONTAINER_NAME=dae:v1


docker run -it  -v ${SRC_ROOT_DIR}:/current_project  -v ${DATA_ROOT_DIR}:/current_project/data ${DOCKER_CONTAINER_NAME}  \
    python /current_project/src/denoising_autoencoder/reduce_discretized_density.py \
    --input_pickle_glob='/current_project/data/p_part_*.pkl' \
    --output_pickle_path=/current_project/data/p_reduced.pkl


#ls /current_project/src

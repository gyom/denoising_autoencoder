#!/bin/bash

export DATA_ROOT_DIR=/Volumes/LaCie/Dropbox/recent/machine_learning/denoising_autoencoders/2019_spiral
export SRC_ROOT_DIR=${HOME}/Documents/code/denoising_autoencoders/2019_spiral

# Don't do that. It doesn't do anything because the absolute reference doesn't translate.
# export PYTHONPATH=${PYTHONPATH}:${SRC_ROOT_DIR}/src

# Not needed for this task.
# -v /Volumes/LaCie/Dropbox/recent/machine_learning/RL2017_deepmindprep/experiments:/current_project/experiments
# -v /Volumes/LaCie/Dropbox/recent/machine_learning/RL2017_deepmindprep/checkpoints:/current_project/checkpoints

docker run -it  -v ${SRC_ROOT_DIR}:/current_project  -v ${DATA_ROOT_DIR}:/current_project/data  tensorflow/tensorflow  python /current_project/src/denoising_autoencoders/script01.py

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


# docker run -it  -v ${SRC_ROOT_DIR}:/current_project  -v ${DATA_ROOT_DIR}:/current_project/data ${DOCKER_CONTAINER_NAME}  \
#    python /current_project/src/denoising_autoencoder/generate_discretized_density.py \
#    --grid_nbr_points=1000 \
#    --grid_radius=1.0 \
#    --output_pickle_path=/current_project/data/p_part_00.pkl


for i in `seq 0 39`
do
    # The actual spiral.
    docker run -it  -v ${SRC_ROOT_DIR}:/current_project  -v ${DATA_ROOT_DIR}:/current_project/data ${DOCKER_CONTAINER_NAME}  \
        python /current_project/src/denoising_autoencoder/generate_discretized_density.py \
        --grid_nbr_points=1000 \
        --grid_radius=1.0 \
        --spiral_noise_sigma=0.01 \
        --nbr_iter=25 \
        --output_pickle_path=/current_project/data/p_part_0${i}.pkl

    # echo "Done with p part ${i}."

    # A sampling proposal q with a lot more noise.
    docker run -it  -v ${SRC_ROOT_DIR}:/current_project  -v ${DATA_ROOT_DIR}:/current_project/data ${DOCKER_CONTAINER_NAME}  \
        python /current_project/src/denoising_autoencoder/generate_discretized_density.py \
        --grid_nbr_points=1000 \
        --grid_radius=1.0 \
        --spiral_noise_sigma=0.1 \
        --nbr_iter=25 \
        --output_pickle_path=/current_project/data/q_part_0${i}.pkl

    # echo "Done with q part ${i}."

done


#ls /current_project/src

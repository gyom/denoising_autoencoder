#!/bin/bash
export N_SAMPLES="10000"
export THINNING_FACTOR="100"


python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --langevin_lambda=0.01 --mcmc_method=langevin --dataset=butterfly --output_dir_prefix=${HOME}/Documents/tmp/2013_02_12_KL_measurements_butterfly --reference_pickled_samples_for_KL=butterfly_samples_n_10000.pkl  | egrep "ratio|divergence" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/run_once/butterfly_checking_KL/2013_02_12_KL_measurements_butterfly_langevin.txt

python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --langevin_lambda=0.01 --mcmc_method=metropolis_hastings_langevin_grad_E --dataset=butterfly --output_dir_prefix=${HOME}/Documents/tmp/2013_02_12_KL_measurements_butterfly --reference_pickled_samples_for_KL=butterfly_samples_n_10000.pkl  | egrep "ratio|divergence" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/run_once/butterfly_checking_KL/2013_02_12_KL_measurements_butterfly_metropolis_hastings_langevin_grad_E.txt

python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --langevin_lambda=0.01 --mcmc_method=metropolis_hastings_langevin_E --dataset=butterfly --output_dir_prefix=${HOME}/Documents/tmp/2013_02_12_KL_measurements_butterfly --reference_pickled_samples_for_KL=butterfly_samples_n_10000.pkl  | egrep "ratio|divergence" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/run_once/butterfly_checking_KL/2013_02_12_KL_measurements_butterfly_metropolis_hastings_langevin_E.txt


python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --proposal_stddev=0.1 --mcmc_method=metropolis_hastings_E --dataset=butterfly --output_dir_prefix=${HOME}/Documents/tmp/2013_02_12_KL_measurements_butterfly --reference_pickled_samples_for_KL=butterfly_samples_n_10000.pkl | egrep "ratio|divergence" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/run_once/butterfly_checking_KL/2013_02_12_KL_measurements_butterfly_metropolis_hastings_E.txt

python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --proposal_stddev=0.1 --mcmc_method=metropolis_hastings_grad_E --dataset=butterfly --output_dir_prefix=${HOME}/Documents/tmp/2013_02_12_KL_measurements_butterfly --reference_pickled_samples_for_KL=butterfly_samples_n_10000.pkl | egrep "ratio|divergence" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/run_once/butterfly_checking_KL/2013_02_12_KL_measurements_butterfly_metropolis_hastings_grad_E.txt



#!/bin/bash
export N_SAMPLES="10000"
export THINNING_FACTOR="100"

python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --langevin_lambda=0.01 --mcmc_method=langevin --dataset=ninja_star --output_dir_prefix=${HOME}/Documents/tmp/2013_02_09_KL_measurements --reference_pickled_samples_for_KL=ninja_star_metropolis_hasting_samples_n_10000_thin_10000_prop_stddev_0.100000.pkl --omit_ninja_star_from_plots | grep "KL divergence value" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/junk/2013_02_09_KL_measurements_langevin.txt

python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --langevin_lambda=0.01 --mcmc_method=metropolis_hastings_langevin_grad_E --dataset=ninja_star --output_dir_prefix=${HOME}/Documents/tmp/2013_02_09_KL_measurements --reference_pickled_samples_for_KL=ninja_star_metropolis_hasting_samples_n_10000_thin_10000_prop_stddev_0.100000.pkl --omit_ninja_star_from_plots | grep "KL divergence value" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/junk/2013_02_09_KL_measurements_metropolis_hastings_langevin_grad_E.txt

python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --langevin_lambda=0.01 --mcmc_method=metropolis_hastings_langevin_E --dataset=ninja_star --output_dir_prefix=${HOME}/Documents/tmp/2013_02_09_KL_measurements --reference_pickled_samples_for_KL=ninja_star_metropolis_hasting_samples_n_10000_thin_10000_prop_stddev_0.100000.pkl --omit_ninja_star_from_plots | grep "KL divergence value" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/junk/2013_02_09_KL_measurements_metropolis_hastings_langevin_E.txt


python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --proposal_stddev=0.1 --mcmc_method=metropolis_hastings_E --dataset=ninja_star --output_dir_prefix=${HOME}/Documents/tmp/2013_02_09_KL_measurements --reference_pickled_samples_for_KL=ninja_star_metropolis_hasting_samples_n_10000_thin_10000_prop_stddev_0.100000.pkl --omit_ninja_star_from_plots | grep "KL divergence value" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/junk/2013_02_09_KL_measurements_metropolis_hastings_E.txt

python script_1.py --n_samples=${N_SAMPLES} --thinning_factor=${THINNING_FACTOR} --proposal_stddev=0.1 --mcmc_method=metropolis_hastings_grad_E --dataset=ninja_star --output_dir_prefix=${HOME}/Documents/tmp/2013_02_09_KL_measurements --reference_pickled_samples_for_KL=ninja_star_metropolis_hasting_samples_n_10000_thin_10000_prop_stddev_0.100000.pkl --omit_ninja_star_from_plots | grep "KL divergence value" >> ${HOME}/umontreal/denoising_autoencoder/mcmc_pof/junk/2013_02_09_KL_measurements_metropolis_hastings_grad_E.txt

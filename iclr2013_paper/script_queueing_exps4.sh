

export EXPNUMBER="852496"
python dae_langevin_sampling.py $HOME/umontreal/denoising_autoencoder/plots/experiment_$EXPNUMBER/results.pkl $HOME/Documents/tmp/sampling_dae/${EXPNUMBER}_JJTJJT_h_1 '{"langevin_a":1.0, "noise_method":"JJTJJT_h", "n_iter":100, "n_sub_iter":100, "n_simulated_samples":10}'

export EXPNUMBER="852496"
python dae_langevin_sampling.py $HOME/umontreal/denoising_autoencoder/plots/experiment_$EXPNUMBER/results.pkl $HOME/Documents/tmp/sampling_dae/${EXPNUMBER}_JJTJJT_h_2 '{"langevin_a":1.0, "noise_method":"JJTJJT_h", "n_iter":100, "n_sub_iter":100, "n_simulated_samples":10}'

export EXPNUMBER="852496"
python dae_langevin_sampling.py $HOME/umontreal/denoising_autoencoder/plots/experiment_$EXPNUMBER/results.pkl $HOME/Documents/tmp/sampling_dae/${EXPNUMBER}_JJTJJT_h_3 '{"langevin_a":1.0, "noise_method":"JJTJJT_h", "n_iter":100, "n_sub_iter":100, "n_simulated_samples":100}'

export EXPNUMBER="852496"
python dae_langevin_sampling.py $HOME/umontreal/denoising_autoencoder/plots/experiment_$EXPNUMBER/results.pkl $HOME/Documents/tmp/sampling_dae/${EXPNUMBER}_JJTJJT_h_4 '{"langevin_a":1.0, "noise_method":"JJTJJT_h", "n_iter":100, "n_sub_iter":100, "n_simulated_samples":100}'






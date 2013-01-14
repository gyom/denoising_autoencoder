
export DENOISING_REPO=/u/alaingui/umontreal/;


#### with 10 000 iterations


python dae_gradient_descent_script.py '{"n_hiddens":2000, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000, "n_spiral_original_samples":1000}'

python dae_gradient_descent_script.py '{"n_hiddens":1000, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000, "n_spiral_original_samples":1000}'


python dae_gradient_descent_script.py '{"n_hiddens":2000, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000, "n_spiral_original_samples":5000}'

python dae_gradient_descent_script.py '{"n_hiddens":1000, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000, "n_spiral_original_samples":5000}'



python dae_gradient_descent_script.py '{"n_hiddens":2000, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.1, "maxiter":10000, "n_spiral_original_samples":5000}'

python dae_gradient_descent_script.py '{"n_hiddens":2000, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":true, "train_noise_stddev":0.01, "maxiter":10000, "n_spiral_original_samples":5000}'



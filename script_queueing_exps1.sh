
export DENOISING_REPO=/u/alaingui/umontreal/;

python dae_gradient_descent_script.py '{"n_hiddens":250, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":1000}'

python dae_gradient_descent_script.py '{"n_hiddens":500, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":1000}'

python dae_gradient_descent_script.py '{"n_hiddens":1000, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":1000}'



python dae_gradient_descent_script.py '{"n_hiddens":250, "spiral_samples_noise_stddev":0.01, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":1000}'

python dae_gradient_descent_script.py '{"n_hiddens":500, "spiral_samples_noise_stddev":0.01, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":1000}'

python dae_gradient_descent_script.py '{"n_hiddens":1000, "spiral_samples_noise_stddev":0.01, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":1000}'


#### with 10 000 iterations


python dae_gradient_descent_script.py '{"n_hiddens":250, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000}'

python dae_gradient_descent_script.py '{"n_hiddens":500, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000}'

python dae_gradient_descent_script.py '{"n_hiddens":1000, "spiral_samples_noise_stddev":0.0, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000}'



python dae_gradient_descent_script.py '{"n_hiddens":250, "spiral_samples_noise_stddev":0.01, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000}'


python dae_gradient_descent_script.py '{"n_hiddens":500, "spiral_samples_noise_stddev":0.01, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000}'

python dae_gradient_descent_script.py '{"n_hiddens":1000, "spiral_samples_noise_stddev":0.01, "want_rare_large_noise":false, "train_noise_stddev":0.01, "maxiter":10000}'



export DAEROOT=/data/lisatmp2/alaingui/dae

python generate_dataset_gaussian_process_manifolds.py --d=10 --n_train=1000 --n_test=1000 --output_dir="${DAEROOT}/datasets/gaussian_process//small_d10"



python train_dae.py --n_hiddens=100 --maxiter=1000 --lbfgs_rank=5 --act_func='["tanh", "tanh"]' --noise_stddevs='[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]' --train_samples_pickle="${DAEROOT}/datasets/gaussian_process/small_d10/train_samples.pkl" --output_dir="${DAEROOT}/dae_trained_models/gp_0001"

python train_dae.py --n_hiddens=100 --maxiter=1000 --lbfgs_rank=5 --act_func='["tanh", "tanh"]' --noise_stddevs='[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]' --train_samples_pickle="${DAEROOT}/datasets/gaussian_process/small_d10/train_samples.pkl" --output_dir="${DAEROOT}/dae_trained_models/gp_0002"





python train_dae.py --n_hiddens=100 --maxiter=100 --lbfgs_rank=5 --act_func='["tanh", "sigmoid"]' --noise_stddevs='[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0001]' --train_samples_pickle="${DAEROOT}/datasets/mnist/train.pkl" --output_dir="${DAEROOT}/dae_trained_models/mnist_0001"



/data/lisatmp2/alaingui/dae/datasets/gaussian_process
/data/lisatmp2/alaingui/dae/datasets/mnist


E = cPickle.load(open("/data/lisa/data/mnist/mnist.pkl"))
cPickle.dump(E[0][0], open("/data/lisatmp2/alaingui/dae/datasets/mnist/train.pkl", "w"))
cPickle.dump(E[0][1], open("/data/lisatmp2/alaingui/dae/datasets/mnist/train_indices.pkl", "w"))
cPickle.dump(E[1][0], open("/data/lisatmp2/alaingui/dae/datasets/mnist/valid.pkl", "w"))
cPickle.dump(E[1][1], open("/data/lisatmp2/alaingui/dae/datasets/mnist/valid_indices.pkl", "w"))
cPickle.dump(E[2][0], open("/data/lisatmp2/alaingui/dae/datasets/mnist/test.pkl", "w"))
cPickle.dump(E[2][1], open("/data/lisatmp2/alaingui/dae/datasets/mnist/test_indices.pkl", "w"))




python generate_dataset_gaussian_mixture_manifold_2.py --d=10 --n_train=1000 --n_test=1000 --mixing_prop=0.5 --leading_eigenvalue=10.0 --n_components=25 --output_dir="/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture"


python generate_dataset_gaussian_mixture_manifold.py --d=4 --n_train=1000 --n_test=1000 --ratio_eigvals=10.0 --n_components=25 --output_dir="/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/12"
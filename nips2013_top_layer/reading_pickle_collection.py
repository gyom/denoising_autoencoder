
import subprocess
import cPickle

# pseudo-parameter
#dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d10/experiment_09"
#dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/gaussian_mixture_d100/experiment_01"

dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/mnist_yann_H1/experiment_03_yann_mnist_H1"
list_files = sorted([e for e in subprocess.check_output("find %s -name extra_details.pkl" % (dir,), shell=True).split("\n") if len(e)>0])

#dir = "/data/lisatmp2/alaingui/dae/generated_samples/experiment_11_03"
#list_files = sorted([e for e in subprocess.check_output("find %s -name sampling_extra_details.pkl" % (dir,), shell=True).split("\n") if len(e)>0])


list_contents = [cPickle.load(open(f)) for f in list_files]

best_ind = [10, 11, 19, 9, 27, 28, 0, 37, 18, 1]
for i in best_ind:
    print list_contents[i]



#[e for e in list_contents if (e['maxiter'] == 1000) and (e['act_func'] == ['tanh', 'tanh'])]

A = np.array([min(*e['model_losses']['gentle_valid']) for e in list_contents])
#A = np.array([min(*e['model_losses']['train']) for e in list_contents])
#A = np.array([min(*e['model_losses']['valid']) for e in list_contents])
for key in ['n_hiddens', 'maxiter', 'lbfgs_rank']:
    print ""
    print "--------------------------------------------------------"
    print key
    print np.array([list_contents[i][key] for i in np.argsort(A)])
    print "--------------------------------------------------------"



A = np.array([min(*e['model_losses']['train']) for e in list_contents])
B = np.array([min(*e['model_losses']['valid']) for e in list_contents])

#iB = np.argsort(B)
#iA = np.argsort(A)
#np.hstack((A[iA].reshape((-1,1)), B[iA].reshape((-1,1))))
#np.hstack((A[iB].reshape((-1,1)), B[iB].reshape((-1,1))))

# This should get us the best index.
# Manual inspection of A,B suggests this index.
# Visual inspected too.
# Now the log also suggests this.
iAB = np.argsort(np.log(A) + np.log(B))

#sorted(A)
#i = np.argmin(A)
#list_abs_path_files[i]


for key in ['n_hiddens', 'maxiter', 'act_func']:
    print ""
    print "--------------------------------------------------------"
    print key
    print np.array([list_contents[i][key] for i in np.argsort(A)])
    print "--------------------------------------------------------"


np.argsort(A)[0:10]
Out[134]: array([109,  15, 118,  85,  32,  35, 114, 116,  28,   7])


np.array([list_contents[i]['n_hiddens'] for i in np.argsort(A)])

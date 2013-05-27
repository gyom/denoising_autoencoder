
import subprocess
import os
import json

root_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/mnist_yann_H1"

#subdirs = ["experiment_05_yann_mnist_H1_normal", "experiment_06_yann_mnist_H1_normal", "experiment_07_yann_mnist_H1_normal",
#           "experiment_05_yann_mnist_H1_kicking", "experiment_06_yann_mnist_H1_kicking", "experiment_07_yann_mnist_H1_kicking",
#           "experiment_05_yann_mnist_H1_walkback", "experiment_06_yann_mnist_H1_walkback", "experiment_07_yann_mnist_H1_walkback"]

#subdirs = ["experiment_05_yann_mnist_H1_walkback", "experiment_06_yann_mnist_H1_walkback", "experiment_07_yann_mnist_H1_walkback"]
subdirs = ["experiment_08_yann_mnist_H1_walkback2"]

loss_names = ["train", "valid", "gentle_valid", "valid_kicking", "valid_walkback"]

def process_subdir(subdir):
    full_dir = os.path.join(root_dir, subdir)
    json_files = [e for e in subprocess.check_output("find %s -name extra_details.json" % (full_dir,), shell=True).split("\n") if len(e) > 0]

    results = dict([(key, []) for key in loss_names])
    for json_file in json_files:
        details = json.load(open(json_file, "r"))
        for loss_name in loss_names:
            results[loss_name].append((json_file, details["model_losses"][loss_name][-1]))

    return results

def concat_dict(D1, D2):
    D = dict(D1.items())
    for (k,v) in D2.items():
        if D.has_key(k):
            D[k] = D[k] + D2[k]
        else:
            D[k] = D2[k]
    return D

def process_subdir_list(L):
    return reduce(concat_dict, [process_subdir(e) for e in L], {})

loss_name = "valid"

results = process_subdir_list(subdirs)[loss_name]

results = sorted(results, key=lambda tup: tup[1])

for r in results[0:5]:
    print r


#for (json_file,_) in results[0:100]:
#    details = json.load(open(json_file, "r"))
#    print details["n_hiddens"]


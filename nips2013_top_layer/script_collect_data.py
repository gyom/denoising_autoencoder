

import subprocess
import os
import json

root_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/mnist_yann_H1"

#subdirs = {"normal":["experiment_05_yann_mnist_H1_normal", "experiment_06_yann_mnist_H1_normal"],
#           "kicking":["experiment_05_yann_mnist_H1_kicking", "experiment_06_yann_mnist_H1_kicking"],
#           "walkback":["experiment_05_yann_mnist_H1_walkback", "experiment_06_yann_mnist_H1_walkback"]}

#subdirs = {"normal":["experiment_07_yann_mnist_H1_normal"],
#           "kicking":["experiment_07_yann_mnist_H1_kicking"],
#           "walkback":["experiment_07_yann_mnist_H1_walkback"]}


subdirs = {"walkback":["experiment_08_yann_mnist_H1_walkback"],
           "walkback2":["experiment_08_yann_mnist_H1_walkback2"]}


loss_names = ["train", "valid", "gentle_valid", "valid_kicking", "valid_walkback"]

def mean(L):
    return 1.0 * sum(L) / len(L)

def process_subdir(subdir):
    full_dir = os.path.join(root_dir, subdir)
    json_files = [e for e in subprocess.check_output("find %s -name extra_details.json" % (full_dir,), shell=True).split("\n") if len(e) > 0]

    results = dict([(key, []) for key in loss_names])
    for json_file in json_files:
        details = json.load(open(json_file, "r"))
        for loss_name in loss_names:
            results[loss_name].append(details["model_losses"][loss_name][-1])

    return results

def concat_dict(D1, D2):
    D = dict(D1.items())
    for (k,v) in D2.items():
        if D.has_key(k):
            D[k] = D[k] + D2[k]
        else:
            D[k] = D2[k]
    return D

def process_subdir_list(L, summary = None):

    results = reduce(concat_dict, [process_subdir(e) for e in L], {})

    if summary is None:
        return results
    else:
        return dict([(key, summary(value)) for (key, value) in results.items()])



#summary = mean
summary = min

results = dict([(key, process_subdir_list(subdirs[key], summary)) for key in subdirs.keys()])

print results
#print process_subdir(subdirs["normal"][0])
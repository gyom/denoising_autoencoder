
import subprocess
import os
import json

import numpy as np

root_dir = "/data/lisatmp2/alaingui/dae/dae_trained_models/mnist_yann_H1"
full_dir = os.path.join(root_dir, "experiment_09_yann_mnist_H1_walkback")

json_files = [e for e in subprocess.check_output("find %s -name extra_details.json" % (full_dir,), shell=True).split("\n") if len(e) > 0]

results = []
for json_file in json_files:
    details = json.load(open(json_file, "r"))
    results.append(details)

results_tagged = [dict({"walkback_param_p": r["noise_stddevs"]["train"][-1]["walkback_param_p"],
                        "n_hiddens":r["n_hiddens"],
                        "maxiter":r["maxiter"] }.items() + r.items()) for r in results]

# We assume that only one sequence of "sampled" values were used during the
# training of all the elements compared. We might as well test that hypothesis
# for cheap while we have the data in hand.
train_sampled_used = [e["sampled"] for e in results[0]["noise_stddevs"]["train"]]
for r in results:
    assert train_sampled_used == [e["sampled"] for e in r["noise_stddevs"]["train"]]




def f(L):

    def m(A):
        return np.hstack([np.array(a).reshape((-1,1)) for a in A]).mean(axis=1)

    def c(A):
        return [np.cov(row) for row in np.hstack([np.array(a).reshape((-1,1)) for a in A])]

    #{"valid_at_0.1":{"mean": m([e["model_losses"]["valid_at_0.1"] for e in L]),
    #                  "cov": c([e["model_losses"]["valid_at_0.1"] for e in L])},
    # ...        }

    #print L[0]["model_losses"].keys()

    return dict([(key, 
                  {"mean": m([e["model_losses"][key] for e in L]),
                   "cov": c([e["model_losses"][key] for e in L])})
                 for key in ["train", "valid", "valid_at_1.0", "valid_at_0.1", "valid_at_0.01"]])


L_n_hiddens = [128, 256, 512]
L_walkback_param_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
summarized_results = []
for n_hiddens in L_n_hiddens:
    for walkback_param_p in L_walkback_param_p:

        # to catch the strange case where one float would not match another float (e.g. 0.009999 and 0.01)

        L = [r for r in results_tagged
                    if (r["n_hiddens"] == n_hiddens and r["walkback_param_p"] == walkback_param_p)]

        print "(n_hiddens, walkback_param_p) == (%d,%f) has %d entries" % (n_hiddens, walkback_param_p, len(L))

        assert len(L) > 0
        summarized_results.append( dict({"n_hiddens" : n_hiddens,
                                         "walkback_param_p" : walkback_param_p}.items() +
                                        f(L).items())  )


#print summarized_results
#quit()


import matplotlib
import pylab
import matplotlib.pyplot as plt

pylab.hold(True)

#L_n_hiddens = [128,256,512]

L_walkback_param_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# start at orange, end at blue
color_start = (255.0 / 255 , 142.0 / 255,   0.0 / 255)
color_end   = (  0.0 / 255 , 100.0 / 255, 204.0 / 255)
colors = [( (1-t)*color_start[0] + t*color_end[0],
            (1-t)*color_start[1] + t*color_end[1],
            (1-t)*color_start[2] + t*color_end[2]  ) for t in np.linspace(0,1,len(L_walkback_param_p))]

#colors = ["#000000"] * 10
assert len(colors) == len(L_walkback_param_p)

#print "---------------"
#print summarized_results[0]
#print "---------------"

print [e["n_hiddens"] for e in summarized_results]
print [e["walkback_param_p"] for e in summarized_results]


output_dir = "/u/alaingui/umontreal/tmp"
for loss_name in ["train", "valid", "valid_at_1.0", "valid_at_0.1", "valid_at_0.01"]:
    for n_hiddens in L_n_hiddens:
        output_image_file = os.path.join(output_dir, "walkback_comparison_loss_%s_n_hiddens%d.png" % (loss_name, n_hiddens,))

        for (walkback_param_p,i) in zip(L_walkback_param_p, range(len(L_walkback_param_p))):



            #loss_name = "valid"
            e = [e for e in summarized_results if (e["n_hiddens"] == n_hiddens) and (e["walkback_param_p"] == walkback_param_p)][0]
            pylab.plot(np.log(np.array(train_sampled_used)), np.log(e[loss_name]['mean']), color=colors[i], linewidth=2)


        pylab.draw()
        pylab.savefig(output_image_file, dpi=150)
        print "Wrote %s" % (output_image_file,)
        pylab.close()


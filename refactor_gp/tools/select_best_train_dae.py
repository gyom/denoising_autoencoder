#!/bin/env python

import os, sys
import json
import shutil
import numpy as np

def main(argv):
    """
    Takes a list of directories with the last one being the location where
    the best model will be put. This is in the same spirit as the "cp" command
    which takes an aribitrary number of arguments with the last one as target.

    This assume that the directories all have this file "extra_details.json"
    which contains a dictionary with the keys

        "post_valid_model_losses"
        "post_alt_valid_model_losses"
        "train_model_losses"
        "valid_model_losses"

    """

    assert len(argv) >= 3
    output_dir = argv[-1]
    source_dirs = argv[1:-1]
    for e in source_dirs:
        assert os.path.exists(e)

    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)

    L = [json.load(open(os.path.join(source_dir, "extra_details.json"), "r")) for source_dir in source_dirs]

    res = np.array( [np.log(e['post_valid_model_losses'][-1]) + np.log(np.array(e['post_alt_valid_model_losses'])).sum() for e in L] )

    print "=== Results for trained DAE considered ==="
    print res
    print ""

    min_index = np.argmin(res)
    best_dir = source_dirs[min_index]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print "Removed %s because it already existed." % (output_dir,)

    print "Copying %s over to %s" % (best_dir, output_dir)
    # shutil.copytree(best_dir, output_dir)


    # Delete the other directories.
    #for source_dir in source_dirs:
    #    shutil.rmtree(source_dir)
    #    print "Deleted %s" % (source_dir,)



if __name__ == "__main__":
    main(sys.argv)

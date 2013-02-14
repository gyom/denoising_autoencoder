#!/bin/env python

import cPickle
import os, sys

def main():
    """
    Explores a pkl file and navigates through nested dictionaries.
    """

    import os, sys

    assert len(sys.argv) >= 2

    pkl_file = sys.argv[1]
    assert os.path.exists(pkl_file)

    if len(sys.argv) >= 3:
        keys = sys.argv[2:]
    else:
        keys = []

    contents = cPickle.load(open(pkl_file, "r"))

    # drive down
    for key in keys:
        assert contents.has_key(key)
        contents = contents[key]

    if type(contents) == type({}):
        #print contents.keys()
        print contents
    #if type(contents) == np.ndarray:
    #    print contents
    else:
        print contents


if __name__ == "__main__":
    main()


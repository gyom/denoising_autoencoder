
import numpy as np
import os
import cPickle

directory = "/data/lisatmp2/alaingui/mnist/yann"

# yann_test_H1.pkl     yann_test_H0.pkl      yann_test_rH0.pkl    yann_test_X.pkl       yann_test_rX.pkl
# yann_train_H1.pkl    yann_train_rX.pkl     yann_train_H0.pkl    yann_train_rH0.pkl    yann_train_X.pkl  
# yann_valid_H0.pkl    yann_valid_rH0.pkl    yann_valid_X.pkl     yann_valid_H1.pkl     yann_valid_rX.pkl 

for phase in ['train', 'test', 'valid']:
    for layer in ['X', 'H0', 'H1', 'rH0', 'rX']:
        filename = os.path.join(directory, "yann_%s_%s.pkl" % (phase, layer))
        data = cPickle.load(open(filename, "r"))

        data_float32 = np.float32(data)
        output_float32_filename = os.path.join(directory, "yann_float32_%s_%s.pkl" % (phase, layer))
        cPickle.dump(data_float32, open(output_float32_filename, "w"))
        print "Wrote %s." % (output_float32_filename,)

        data_float16 = np.float16(data)
        output_float16_filename = os.path.join(directory, "yann_float16_%s_%s.pkl" % (phase, layer))
        cPickle.dump(data_float16, open(output_float16_filename, "w"))
        print "Wrote %s." % (output_float16_filename,)


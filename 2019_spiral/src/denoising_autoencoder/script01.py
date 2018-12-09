
"""

"""

import sys
sys.path.insert(0,'/current_project/src')

import tensorflow as tf

import denoising_autoencoders
# from denoising_autoencoders.tools_tfrecords_catsanddogs import run



flags = tf.flags
logging = tf.logging

#flags.DEFINE_string("task", 0,
#                    "Task id. Usually coming from the Borg scheduler.")
flags.DEFINE_string("hparams", "",
                    "Comma separated list of name=value hyperparameter pairs.")
flags.DEFINE_string(
    "train_dir", None,
    "Where to automatically save the model checkpoints during training.")

FLAGS = flags.FLAGS


def run():

def run():

    hparams = tf.contrib.training.HParams(
        num_preprocess_threads=4,
        batch_size=64,
        #summary_writer_dir="train_dir_fresh",
        train_dir="train_dir_fresh",
        nbr_steps=10000,
        #restore_path="", # has to be the exact file (e.g. "check.ckpt-800")
        #
        gpu_percent=0.45,
        #
        ### learning rates ###
        learning_rate=1e-4,
        learning_rate_decay_steps=50000,
        learning_rate_decay_rate=0.90,
        ### rmsprop ###
        rmsprop_decay=0.9,
        rmsprop_momentum=0.9,
        #
        # mnist_train_record_path="/Users/gyomalin/Documents/ML_data/mnist_records/mnist_train_record",
        # mnist_valid_record_path="/Users/gyomalin/Documents/ML_data/mnist_records/mnist_valid_record",
        # mnist_test_record_path="/Users/gyomalin/Documents/ML_data/mnist_records/mnist_test_record",
        # mnist_train_record_path=os.path.join(os.environ['HOME'], "Documents/ML_data/mnist_records/mnist_train_record"),
        # mnist_valid_record_path=os.path.join(os.environ['HOME'], "Documents/ML_data/mnist_records/mnist_valid_record"),
        # mnist_test_record_path=os.path.join(os.environ['HOME'], "Documents/ML_data/mnist_records/mnist_test_record"),
        #
        alpha_scale=10.0
    )
    if FLAGS.hparams:
        hparams.parse(FLAGS.hparams)




if __name__ == "__main__":
  run()

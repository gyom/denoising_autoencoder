#!/usr/bin/env python
# encoding: utf-8
"""
dae.py

Guillaume Alain.
"""

import sys
import os
import pdb
import numpy as np

from theano import *
import theano.tensor as T

class DAE(object):

    def __init__(self):
        pass
    
    def encode(self, X):
        error("Abstract method")

    def decode(self, H):
        error("Abstract method")

    def encode_decode(self, X):
        error("Abstract method")

    def model_loss(self, X, noisy_X):
        error("Abstract method")


def main():
    pass


if __name__ == '__main__':
    main()

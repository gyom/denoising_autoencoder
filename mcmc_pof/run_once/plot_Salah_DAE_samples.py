

import numpy as np
import cPickle
import yann_dauphin_utils
import PIL

image_output_file = "/u/alaingui/Documents/tmp/Salah_DAE_2013_02_06/metropolis_hastings_langevin_grad_E/1360188911/digits.png"

samples_pkl_file = "/u/alaingui/Documents/tmp/Salah_DAE_2013_02_06/metropolis_hastings_langevin_grad_E/1360188911/samples.pkl"
samples = cPickle.load(open(samples_pkl_file,"r"))

assert len(samples.shape) == 2
N = samples.shape[0]
n_inputs = samples.shape[1]


tile_j = int(np.ceil(np.sqrt(N)))
tile_i = int(np.ceil(float(N) / tile_j))

img_j = int(np.ceil(np.sqrt(n_inputs)))
img_i = int(np.ceil(float(n_inputs) / img_j))

from PIL import Image
image = Image.fromarray(yann_dauphin_utils.tile_raster_images(
         X = samples,
         img_shape = (img_j,img_i), tile_shape = (tile_j, tile_i),
         tile_spacing=(1,1)))

image.save(image_output_file)
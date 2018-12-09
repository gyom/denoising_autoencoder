from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.testing as npt

def convert_instant_rewards_to_discounted_accumulated_rewards(instant_rewards, gamma=1.0):
  L_reversed_G = []
  L_reversed_r = reversed(list(instant_rewards))

  acc_G = 0.0
  for r in L_reversed_r:
    current_G = r + gamma * acc_G
    L_reversed_G.append(current_G)
    acc_G = current_G

  discounted_accumulated_rewards = np.array(list(reversed(L_reversed_G)))
  return discounted_accumulated_rewards




def test_convert_instant_rewards_to_discounted_accumulated_rewards():

  instant_rewards = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
  ref_discounted_accumulated_rewards = np.array([0.5*(1.0+0.5*1.25), 1.0+0.5*1.25 , 1.25, 0.5, 1.0])

  discounted_accumulated_rewards = convert_instant_rewards_to_discounted_accumulated_rewards(instant_rewards, gamma=0.5)
  npt.assert_array_almost_equal(discounted_accumulated_rewards, ref_discounted_accumulated_rewards)

test_convert_instant_rewards_to_discounted_accumulated_rewards()

# Method to calculate the Flory Huggins Free Energy
# Only for 2 components currently

import sys
sys.path.insert(0, "../")

from constants import constants

import numpy as np
from numba import jit


@jit
def floryHuggins_2c(phi:constants.DTYPE_F, chi:constants.DTYPE_F):
	res = phi*np.log(phi) + (1-phi)*np.log(1-phi) + chi*phi*(1-phi)
	return res
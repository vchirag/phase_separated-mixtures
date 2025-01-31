import sys
sys.path.insert(0, "include/")

from include.constants import constants

from src.run_brute_force import run_brute_force
from src.run_walk import run_walk

import numpy as np

if __name__ == "__main__":	

	dof = constants.DTYPE_I(2)
	size = constants.DTYPE_I(100)

	PHI_1_GLOBAL = constants.DTYPE_F(0.666)
	CHI = constants.DTYPE_F(1)

	run_brute_force(dof, size, PHI_1_GLOBAL, CHI)


	beta = 10
	nSteps = 100

	run_walk(dof, size, PHI_1_GLOBAL, CHI, beta, nSteps, 10)

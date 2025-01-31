import sys
sys.path.insert(0, "include/")

from include.constants import constants

from src.run_brute_force import run_brute_force
from src.run_walk import run_walk

import numpy as np

if __name__ == "__main__":	

	dof = constants.DTYPE_I(2)
	size = constants.DTYPE_I(100)

	# PHI_1_GLOBAL = constants.DTYPE_F(0.666)
	# CHI = constants.DTYPE_F(1)

	PHI_1_GLOBALs = np.linspace(1e-3, 1-1e-3, 10)
	CHIs = np.linspace(1, 3, 10)
	
	beta = 10
	nSteps = 1000000

	for PHI_1_GLOBAL in PHI_1_GLOBALs:
		for CHI in CHIs:

			run_brute_force(dof, size, PHI_1_GLOBAL, CHI)

			run_walk(dof, size, PHI_1_GLOBAL, CHI, beta, nSteps, 5)

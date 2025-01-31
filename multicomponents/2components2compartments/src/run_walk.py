import sys
sys.path.insert(0, "../include/")

from constants import constants
from mcmc.mcmc import mcmc

import numpy as np

def run_walk(dof:constants.DTYPE_I, size:constants.DTYPE_I, 
	PHI_1_GLOBAL:constants.DTYPE_F, CHI:constants.DTYPE_F, 
	beta:constants.DTYPE_F, nSteps:constants.DTYPE_I, 
	replicas:constants.DTYPE_I):

	print("Running MCMC Walker!")
	walker = mcmc(dof = dof, size = size, phi_1_global = PHI_1_GLOBAL, chi = CHI, beta = beta)
	
	ctr = 1

	flag = 0

	while(ctr <= replicas):

		start_idx_phi11 = np.random.randint(size)
		start_idx_eta1 = np.random.randint(size)

		flag = walker.simulate(start_idx_phi11, start_idx_eta1, nSteps, saveFlag=True, replica=ctr)

		if flag == 0:
			ctr += 1
		
		else:
			print("False")
			continue

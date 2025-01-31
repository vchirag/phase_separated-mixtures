import sys
sys.path.insert(0, "../include/")

from constants import constants
from mcmc.mcmc import mcmc



def run_walk(dof:constants.DTYPE_I, size:constants.DTYPE_I, 
	PHI_1_GLOBAL:constants.DTYPE_F, CHI:constants.DTYPE_F, 
	beta:constants.DTYPE_F, nSteps:constants.DTYPE_I,
	start_idx_phi11:constants.DTYPE_I, start_idx_eta1:constants.DTYPE_I):

	print("Running MCMC Walker!")
	walker = mcmc(dof = dof, size = size, phi_1_global = PHI_1_GLOBAL, chi = CHI, beta = beta)
	
	# A sanity check here or within the class?! To check if the start idxs are not violating any system constraints 
	walker.simulate(start_idx_phi11, start_idx_eta1, nSteps, saveFlag=True)


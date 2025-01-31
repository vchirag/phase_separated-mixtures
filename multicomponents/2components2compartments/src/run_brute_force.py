import sys
sys.path.insert(0, "../include/")

from constants import constants
from brute_force.brute_force import brute_force


def run_brute_force(dof:constants.DTYPE_I, size:constants.DTYPE_I, PHI_1_GLOBAL:constants.DTYPE_F, CHI:constants.DTYPE_F):

	print("Running Brute Force!")

	brute = brute_force(dof, size, PHI_1_GLOBAL, CHI)
	brute.brute(saveFlag=True)
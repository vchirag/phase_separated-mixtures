import sys
sys.path.insert(0, "include/")

from constants import constants

from free_energy import flory_huggins

from mcmc.mcmc import mcmc

from brute_force.brute_force import brute_force
import numpy as np



print(constants.DTYPE_I)
print(constants.DTYPE_F)
print(constants.EPSILON)

print(flory_huggins.floryHuggins_2c(0.1, 0.1))

size = 100
PHI_1_GLOBAL = 0.666
CHI = 1
beta = 10

walker = mcmc(dof = 2, size = size, phi_1_global = PHI_1_GLOBAL, chi = CHI, beta = beta)
walker.simulate(3, 3, 1000000, saveFlag=True)

# brute = brute_force(2, size, PHI_1_GLOBAL, CHI)
# brute.brute(saveFlag=True)

import os
import sys
sys.path.insert(0, "../")

from constants import constants
from free_energy import flory_huggins

import numpy as np
import pandas as pd
from tqdm import tqdm

class mcmc:
	def __init__(self, dof:constants.DTYPE_I, size:constants.DTYPE_I, phi_1_global:constants.DTYPE_F, chi:constants.DTYPE_F, beta:constants.DTYPE_F):
		self.dof = dof
		self.size = size # Mesh size
		self.epsilon = constants.EPSILON
		self.phi_1_global = phi_1_global
		self.chi = chi
		self.beta = beta # Walker's temperature

	def createMesh(self):
		mesh = np.zeros((self.dof, self.size))

		for row in range(self.dof):
			mesh[row] = np.linspace(self.epsilon, 1 - self.epsilon, self.size)

		return mesh


	def metropolisStep(self, current_idx_phi_11:constants.DTYPE_I, current_idx_eta_1:constants.DTYPE_I):

		mesh = self.createMesh()

		current_phi_12 = (self.phi_1_global - mesh[1][current_idx_eta_1]*mesh[0][current_idx_phi_11])/(1-mesh[1][current_idx_eta_1])

		current_f1 = flory_huggins.floryHuggins_2c(mesh[0][current_idx_phi_11], self.chi)
		current_f2 = flory_huggins.floryHuggins_2c(current_phi_12, self.chi)

		current_f = mesh[1][current_idx_eta_1]*current_f1 + (1-mesh[1][current_idx_eta_1])*current_f2

		proposal_idx_phi_11 = 0
		proposal_idx_eta_1 = 0

		# Forcing the walker to move away from the boundaries
		if (current_idx_phi_11 == self.size-1):
			proposal_idx_phi_11 = current_idx_phi_11 - 1
			proposal_idx_eta_1 = current_idx_eta_1

		elif (current_idx_phi_11 == 0):
			proposal_idx_phi_11 = current_idx_phi_11 + 1
			proposal_idx_eta_1 = current_idx_eta_1

		elif (current_idx_eta_1 == self.size-1):
			proposal_idx_phi_11 = current_idx_phi_11
			proposal_idx_eta_1 = current_idx_eta_1 -1

		elif (current_idx_eta_1 == 0):
			proposal_idx_phi_11 = current_idx_phi_11
			proposal_idx_eta_1 = current_idx_eta_1 + 1 

		# Choosing the next move randomly
		else:
			p = np.random.uniform(0, 1)

			if p <= 0.25:
				proposal_idx_phi_11 = current_idx_phi_11 + 1
				proposal_idx_eta_1 = current_idx_eta_1

			elif p <= 0.50:
				proposal_idx_phi_11 = current_idx_phi_11 - 1
				proposal_idx_eta_1 = current_idx_eta_1

			elif p <= 0.75:
				proposal_idx_phi_11 = current_idx_phi_11
				proposal_idx_eta_1 = current_idx_eta_1 + 1

			else:
				proposal_idx_phi_11 = current_idx_phi_11
				proposal_idx_eta_1 = current_idx_eta_1 - 1

		proposal_phi_12 = (self.phi_1_global - mesh[1][proposal_idx_eta_1]*mesh[0][proposal_idx_phi_11])/(1-mesh[1][proposal_idx_eta_1])

		if proposal_phi_12 <= 1-self.epsilon and proposal_phi_12 >= 0+self.epsilon:
			proposal_f1 = flory_huggins.floryHuggins_2c(mesh[0][proposal_idx_phi_11], self.chi)
			proposal_f2 = flory_huggins.floryHuggins_2c(proposal_phi_12, self.chi)

			proposal_f = mesh[1][proposal_idx_eta_1]*proposal_f1 + (1-mesh[1][proposal_idx_eta_1])*proposal_f2

			delta_f = proposal_f - current_f

			if delta_f < 0 or np.random.uniform(0, 1) < np.exp(-self.beta*delta_f):
				return [proposal_idx_phi_11, proposal_idx_eta_1]
			else:
				return [current_idx_phi_11, current_idx_eta_1]
		else:
			return [current_idx_phi_11, current_idx_eta_1]


	def simulate(self, start_idx_phi_11:constants.DTYPE_I, start_idx_eta_1:constants.DTYPE_I, nSteps:constants.DTYPE_I, saveFlag:bool, replica:constants.DTYPE_I):
		walked_idx_phi_11 = []
		walked_idx_eta_1 = []
		walked_phi_11 =[]
		walked_eta_1 = []

		mesh = self.createMesh()

		# Sanity check- if the start idxs are within the constrained region or not
		start_phi_12 = (self.phi_1_global - mesh[1][start_idx_eta_1]*mesh[0][start_idx_phi_11])/(1-mesh[1][start_idx_eta_1])

		if 0 <= start_phi_12 <= 1:
			walked_idx_phi_11.append(start_idx_phi_11)
			walked_phi_11.append(mesh[0][start_idx_phi_11])

			walked_idx_eta_1.append(start_idx_eta_1)
			walked_eta_1.append(mesh[1][start_idx_eta_1])

			for step in tqdm(range(nSteps)):
				# print([start_idx_phi_11, start_idx_eta_1])

				new_idx_phi_11, new_idx_eta_1 = self.metropolisStep(start_idx_phi_11, start_idx_eta_1)
				start_idx_phi_11, start_idx_eta_1 = new_idx_phi_11, new_idx_eta_1

				walked_idx_phi_11.append(start_idx_phi_11)
				walked_phi_11.append(mesh[0][start_idx_phi_11])

				walked_idx_eta_1.append(start_idx_eta_1)
				walked_eta_1.append(mesh[1][start_idx_eta_1])

			if saveFlag:
				df = pd.DataFrame()
				df["idx_phi11"] = walked_idx_phi_11
				df["idx_eta1"] = walked_idx_eta_1
				df["phi11"] = walked_phi_11
				df["eta1"] = walked_eta_1

				output_filepath = f"data/mcmc/mesh-{self.size}/chi-{self.chi:.3f}/phi_g-{self.phi_1_global:.3f}/steps-{nSteps}/beta-{self.beta}/df"
				output_filename = f"df_mcmc-replica{replica}.pkl"

				if not os.path.exists(output_filepath):
					os.makedirs(output_filepath)

				file = os.path.join(output_filepath, output_filename)    
				df.to_pickle(file, compression='gzip')
				print(f"Saved @ {file}")
			return 0

		else:
			return -1
import os
import sys
sys.path.insert(0, "../")

from constants import constants
from free_energy import flory_huggins

import numpy as np
import pandas as pd

class brute_force:
	def __init__(self, dof:constants.DTYPE_I, size:constants.DTYPE_I, phi_1_global:constants.DTYPE_F, chi:constants.DTYPE_F):
		self.dof = dof
		self.size = size # Mesh size
		self.epsilon = constants.EPSILON
		self.phi_1_global = phi_1_global
		self.chi = chi

	def createMesh(self):
		mesh = np.zeros((self.dof, self.size))

		for row in range(self.dof):
			mesh[row] = np.linspace(self.epsilon, 1 - self.epsilon, self.size)

		return mesh

	def brute(self, saveFlag=bool):
		acceptable_idx_phi11s = []
		acceptable_idx_eta1s = []
		acceptable_phi_11s = []
		acceptable_eta_1s = []
		acceptable_fs = []

		mesh = self.createMesh()
		idx_phi_11 = np.arange(len(mesh[0]))
		idx_eta_1 = np.arange(len(mesh[1]))

		for idx_p in idx_phi_11:
			for idx_e in idx_eta_1:
				phi_12 = (self.phi_1_global - mesh[1][idx_e]*mesh[0][idx_p])/(1-mesh[1][idx_e])

				if (0 <= phi_12 <= 1):
					f1 = flory_huggins.floryHuggins_2c(mesh[0][idx_p], self.chi)
					f2 = flory_huggins.floryHuggins_2c(phi_12, self.chi)

					f = mesh[1][idx_e]*f1 + ((1-mesh[1][idx_e])*f2)

					acceptable_idx_phi11s.append(idx_p)
					acceptable_phi_11s.append(mesh[0][idx_p])
					acceptable_idx_eta1s.append(idx_e)
					acceptable_eta_1s.append(mesh[1][idx_e])
					acceptable_fs.append(f)
					# print(mesh[0][idx_p], mesh[1][idx_e], phi_12, f)


		if saveFlag:
			df = pd.DataFrame()
			df["idx_phi11"] = acceptable_idx_phi11s
			df["idx_eta1"] = acceptable_idx_eta1s
			df["phi11"] = acceptable_phi_11s
			df["eta1"] = acceptable_eta_1s
			df["F"] = acceptable_fs

			output_filepath = f"data/brute_force/mesh-{self.size}/chi-{self.chi:.3f}/phi_g-{self.phi_1_global:.3f}/df"
			output_filename = f"df_brute.pkl"

			if not os.path.exists(output_filepath):
			    os.makedirs(output_filepath)

			file = os.path.join(output_filepath, output_filename)    
			df.to_pickle(file, compression='gzip')
			print(f"Saved @ {file}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
from scipy.stats import entropy

# BETAs = [1, 2, 4, 8, 10]
# BETAs = [10]
BETAs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
nSTEPSs = [100, 1000, 10000, 100000, 1000000]
# nSTEPSs = [1000000]
REPLICAs = [1, 2, 3, 4, 5]
PHI_1_GLOBALs = np.linspace(1e-3, 1-1e-3, 10)
# PHI_1_GLOBALs = [0.555]
# CHIs = [2.556]
CHIs = np.linspace(1, 3, 10)

replicas = 5
for phi_g in tqdm(PHI_1_GLOBALs):
    for chi in CHIs:
        for nSteps in nSTEPSs:
            for beta in BETAs:
                for replica in range(1, replicas+1):
                    df_mcmc = pd.read_pickle(f"../../data/mcmc/mesh-100/chi-{chi:.3f}/phi_g-{phi_g:.3f}/steps-{nSteps}/beta-{beta}/df/df_mcmc-replica{replica}.pkl", compression = "gzip")
                    df_counts = df_mcmc.groupby(["phi11", "eta1"]).size().reset_index(name="count")
                    df_counts["probability_P"] = df_counts["count"]/df_counts["count"].sum()
                
                    if replica == 1:
                        merged = df_counts
                    else:
                        merged = pd.concat([merged, df_counts]).groupby(["phi11", "eta1"], as_index=False).sum()
                
                    merged["probability_P"] = merged["count"] / merged["count"].sum()

                # fig, ax = plt.subplots(figsize = (8,6))
                # scatter = ax.scatter(merged["phi11"], merged["eta1"], c = merged["probability_P"], cmap = "viridis", s = 8)
                # # ax.scatter(merged["phi11"], merged["eta1"], c = merged["probability_P"], cmap = "viridis", s=8)
                # cbar = plt.colorbar(scatter, ax=ax, label="Probability")
                
                # # Bolden the label
                # cbar.set_label("Probability", weight='bold', fontsize=12)
                
                # # Bolden the tick numbers
                # cbar.ax.tick_params(axis='y', which='both', width=1.5, length=4, labelsize=12)
                # for label in cbar.ax.get_yticklabels():
                #     label.set_weight('bold')
                # ax.set_xlabel(r"$\mathbf{\phi_{11}}$", fontsize = 12)
                # ax.set_ylabel(r"$\mathbf{\eta_1}$", fontsize = 12)
                # ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, labelsize=12)
                # ax.tick_params(axis='both', which='minor', direction='in', width=1.5, length=4, labelsize=10)
                # for label in ax.get_xticklabels() + ax.get_yticklabels():
                #     label.set_fontweight("bold")
                # fig.tight_layout()



                fig, ax = plt.subplots(figsize = (8, 6))
                scatter = ax.scatter(merged["phi11"], merged["eta1"], c = merged["probability_P"], cmap = "viridis", s = 8)
                    
                # ax.scatter(df_merged["phi11"], df_merged["eta1"], c = df_merged["probability_P"], cmap = "viridis", s = 8)
                # title = "MCMC\n\n" + r"$\beta=$" + f"{beta:.3f}" + "\nSteps = " + f"{nSteps}" + "\nReplica " + f"{replica}" 
                # ax[1].set_title(title)
            
                cbar = plt.colorbar(scatter, ax=ax, label="Probability")
                
                # Bolden the label
                cbar.set_label("Probability", weight='bold', fontsize=14)
                
                # Bolden the tick numbers
                cbar.ax.tick_params(
                    axis='y',          # Apply to y-axis (colorbar's axis)
                    direction='in',    # Ticks point inward
                    length=5,          # Tick length (adjust as needed)
                    width=1.5,         # Tick width (optional)
                    labelsize=10,      # Tick label size
                )
                
                # Bolden the tick labels (numbers)
                for label in cbar.ax.get_yticklabels():
                    label.set_weight('bold')
                    
                ax.set_xlabel(r"$\mathbf{\phi_{11}}$", fontsize = 16)
                ax.set_ylabel(r"$\mathbf{\eta_1}$", fontsize = 16)
                ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, labelsize=12)
                ax.tick_params(axis='both', which='minor', direction='in', width=1.5, length=4, labelsize=10)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight("bold")
                fig.tight_layout()



                
                output_filepath = f"../../analysed_data/chi-{chi:.3f}/phi_g-{phi_g:.3f}/steps-{nSteps}/beta-{beta}/"
                output_filename = f"landscape-merged_simple_average.png"
                
                if not os.path.exists(output_filepath):
                    os.makedirs(output_filepath)
                
                file = os.path.join(output_filepath, output_filename)     
                
                plt.savefig(file, dpi = 400)
                plt.close()


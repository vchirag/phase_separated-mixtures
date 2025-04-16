import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.stats import entropy
def plotter_kl_vs_beta(kl, phi_g, chi, saveFlag):
    plot_labels = [10000, 100000, 1000000]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "s", "d", "^", "."]
    betas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    fig, ax = plt.subplots(figsize = (8, 6))
    ctr = 1

    for idx, _ in enumerate(plot_labels):
        exponent = int(np.log10(plot_labels[idx]))
        # Format label as "10^exponent" (LaTeX-style)
        label = f"nSteps=$\\mathbf{{10^{{{exponent}}}}}$"  # Bold 10^exponent
        ax.plot(betas, kl[idx], label = label, color = colors[idx], marker = markers[idx], linewidth = 3)
        ctr += 1

    # legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10, frameon=True)
    legend = ax.legend(loc='upper right', fontsize=10, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold") 
    
    ax.set_xlabel(r"$\mathbf{\frac{1}{k_BT}}$", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{\langle {KL}\rangle_{{reps}}}$", fontsize=16, fontweight="bold")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
                            
    ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, labelsize=12)
    ax.tick_params(axis='both', which='minor', direction='in', width=1.5, length=4, labelsize=10)
    ax.set_xticks(BETAs)
    ax.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.6)
    plt.yscale("log")
        
    # title = r"$\phi_1^{\text{global}}$ = " + f"{phi_g:.3f}" + "\n" + r"$\chi = $" + f"{chi:.3f}"
    
    # fig.suptitle(title)
    fig.tight_layout()
    
    if saveFlag:
        output_filepath = f"../../analysed_data/chi-{chi:.3f}/phi_g-{phi_g:.3f}"
        output_filename = f"KL-vs-beta.png"

        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)
        
        file = os.path.join(output_filepath, output_filename)     
        
        # plt.savefig(file,  dpi=400, bbox_inches='tight')
        plt.savefig(file,  dpi=400)
        plt.close()

        print(f"Saved @ {file}")

def plotter_kl_vs_steps(kl, phi_g, chi, saveFlag):
    steps = [100, 1000, 10000, 100000, 1000000]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10"]
    markers = ["o", "s", "d", "^", "."]
    # plot_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plot_labels = [1, 2, 4, 8, 10]
    
    fig, ax = plt.subplots(figsize = (8, 6))
    ctr = 1

    for idx, _ in enumerate(plot_labels):
        exponent = int(np.log10(plot_labels[idx]))
        # Format label as "10^exponent" (LaTeX-style)
        label = f"$\\mathbf{{\\frac{{1}}{{k_BT}}}}=\\mathbf{{{plot_labels[idx]}}}$"
        ax.plot(steps, kl[idx], label = label, color = colors[idx], marker= markers[idx], linewidth = 3, alpha = 0.5)
        ctr += 1

    # legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10, frameon=True)
    legend = ax.legend(loc='upper right', fontsize=10, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold") 
    
    ax.set_xlabel("nSteps", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{\langle {KL}\rangle_{{reps}}}$", fontsize=16, fontweight="bold")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
                            
    ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, labelsize=12)
    ax.tick_params(axis='both', which='minor', direction='in', width=1.5, length=4, labelsize=10)
    ax.set_xticks(BETAs)
    ax.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.6)
    plt.xscale("log")
    # plt.yscale("log")
        
    # title = r"$\phi_1^{\text{global}}$ = " + f"{phi_g:.3f}" + "\n" + r"$\chi = $" + f"{chi:.3f}"
    
    # fig.suptitle(title)
    fig.tight_layout()
    
    if saveFlag:
        output_filepath = f"../../analysed_data/chi-{chi:.3f}/phi_g-{phi_g:.3f}"
        output_filename = f"KL-vs-steps.png"

        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)
        
        file = os.path.join(output_filepath, output_filename)     
        
        # plt.savefig(file,  dpi=400, bbox_inches='tight')
        plt.savefig(file,  dpi=400)
        plt.close()

        print(f"Saved @ {file}")


def plotter_kl_vs_steps_loglog(kl, phi_g, chi, saveFlag):
    steps = [100, 1000, 10000, 100000, 1000000]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10"]
    markers = ["o", "s", "d", "^", "."]
    # plot_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plot_labels = [1, 2, 4, 8, 10]
    
    fig, ax = plt.subplots(figsize = (8, 6))
    ctr = 1

    for idx, _ in enumerate(plot_labels):
        exponent = int(np.log10(plot_labels[idx]))
        label = f"$\\mathbf{{\\frac{{1}}{{k_BT}}}}=\\mathbf{{{plot_labels[idx]}}}$"
        ax.plot(steps, kl[idx], label = label, color = colors[idx], marker= markers[idx], linewidth = 3, alpha = 0.5)
        ctr += 1

    # legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10, frameon=True)
    legend = ax.legend(loc='upper right', fontsize=10, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold") 
    
    ax.set_xlabel("nSteps", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{\langle {KL}\rangle_{{reps}}}$", fontsize=16, fontweight="bold")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
                            
    ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, labelsize=12)
    ax.tick_params(axis='both', which='minor', direction='in', width=1.5, length=4, labelsize=10)
    ax.set_xticks(BETAs)
    ax.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
        
    # title = r"$\phi_1^{\text{global}}$ = " + f"{phi_g:.3f}" + "\n" + r"$\chi = $" + f"{chi:.3f}"
    
    # fig.suptitle(title)
    fig.tight_layout()
    
    if saveFlag:
        output_filepath = f"../../analysed_data/chi-{chi:.3f}/phi_g-{phi_g:.3f}"
        output_filename = f"KL-vs-steps_loglog.png"

        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)
        
        file = os.path.join(output_filepath, output_filename)     
        
        # plt.savefig(file,  dpi=400, bbox_inches='tight')
        plt.savefig(file,  dpi=400)
        plt.close()

        print(f"Saved @ {file}")

def plotter_landscapes(df_merged, phi_g, chi, beta, nSteps, replica, saveFlag):
    fig, ax = plt.subplots(figsize = (8, 6))
    scatter = ax.scatter(df_merged["phi11"], df_merged["eta1"], c = df_merged["probability_Q"], cmap = "viridis", s = 8)
    
    # ax.set_title("Brute Force\n\n\n\n")
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
    for label in cbar.ax.get_yticklabels():
        label.set_weight('bold')
    
    ax.set_xlabel(r"$\mathbf{\phi_{11}}$", fontsize = 16)
    ax.set_ylabel(r"$\mathbf{\eta_1}$", fontsize = 16)
    ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, labelsize=12)
    ax.tick_params(axis='both', which='minor', direction='in', width=1.5, length=4, labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()

    if saveFlag:
        output_filepath = f"../../analysed_data/chi-{chi:.3f}/phi_g-{phi_g:.3f}/steps-{nSteps}/beta-{beta}/"
        # output_filename = f"landscape-replica{replica}.png"
        output_filename = f"landscape-brute_force.png"

        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)
        
        file = os.path.join(output_filepath, output_filename)     
        
        plt.savefig(file, dpi = 400)
        plt.close()

        print(f"Saved @ {file}")

    fig, ax = plt.subplots(figsize = (8, 6))
    scatter = ax.scatter(df_merged["phi11"], df_merged["eta1"], c = df_merged["probability_P"], cmap = "viridis", s = 8)
        
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
    for label in cbar.ax.get_yticklabels():
        label.set_weight('bold')
        
    ax.set_xlabel(r"$\mathbf{\phi_{11}}$", fontsize = 16)
    ax.set_ylabel(r"$\mathbf{\eta_1}$", fontsize = 16)
    ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, labelsize=12)
    ax.tick_params(axis='both', which='minor', direction='in', width=1.5, length=4, labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()
    
    if saveFlag:
        output_filepath = f"../../analysed_data/chi-{chi:.3f}/phi_g-{phi_g:.3f}/steps-{nSteps}/beta-{beta}/"
        output_filename = f"landscape-replica{replica}.png"
        # output_filename = f"landscape-brute_force.png"

        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)
        
        file = os.path.join(output_filepath, output_filename)     
        
        plt.savefig(file, dpi = 400)
        plt.close()

        # print(f"Saved @ {file}")
   
    # title = r"$\phi_1^{\text{global}}$ = " + f"{phi_g:.3f}" + "\n" + r"$\chi = $" + f"{chi:.3f}"
        
    # fig.suptitle(title)
    
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

for phi_g in tqdm(PHI_1_GLOBALs):
    for chi in CHIs:
        kl = []
        for beta in BETAs:
            kl_b = []
            for nSteps in nSTEPSs:
                kl_replica = []
                for replica in REPLICAs:
                    df_brute = pd.read_pickle(f"../../data/brute_force/mesh-100/chi-{chi:.3f}/phi_g-{phi_g:.3f}/df/df_brute.pkl", compression = "gzip")
                    
                    df_brute["boltzmann_factor"] = np.exp(-beta*df_brute["F"])
                    Z = df_brute["boltzmann_factor"].sum()
                    df_brute["probability_Q"] = df_brute["boltzmann_factor"]/Z
                    
                    
                    df_mcmc = pd.read_pickle(f"../../data/mcmc/mesh-100/chi-{chi:.3f}/phi_g-{phi_g:.3f}/steps-{nSteps}/beta-{beta}/df/df_mcmc-replica{replica}.pkl", compression = "gzip")
                    df_counts = df_mcmc.groupby(["phi11", "eta1"]).size().reset_index(name="count")
                    df_counts["probability_P"] = df_counts["count"]/df_counts["count"].sum()
                    
                    
                    df_merged = pd.merge(df_counts[['phi11', 'eta1', 'probability_P']], df_brute[['phi11', 'eta1', 'probability_Q']], on=['phi11', 'eta1'], how='outer')
                    df_merged['probability_P'] = df_merged['probability_P'].fillna(0)  # Unvisited states get P(x,y) = 0
                    epsilon = 1e-10
                    df_merged['probability_Q'] = df_merged['probability_Q'].fillna(epsilon)
                    
                    P = df_merged["probability_P"].values
                    Q = df_merged["probability_Q"].values

                    plotter_landscapes(df_merged, phi_g, chi, beta, nSteps, replica, saveFlag=True)
                    
                    KL_divergence = entropy(P, Q)
                    # print(f"phi_g {phi_g:.3f}, chi {chi:.3f}, beta {beta:.3f}, steps {nSteps}, KL Divergence: {KL_divergence:.3f}")
                    
                    kl_replica.append(KL_divergence)
                kl_b.append(np.mean(kl_replica))
            kl.append(kl_b)
        # plotter_kl_vs_steps(kl, phi_g, chi, saveFlag=True)
        # plotter_kl_vs_steps_loglog(kl, phi_g, chi, saveFlag=True)
            # print()
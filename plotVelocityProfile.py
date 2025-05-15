import os
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy as sp
import pandas as pd
import pathlib

plt.style.use(["science", "nature"])

from FoamUtils import file_utils as fu
from FoamUtils import wall_profiles as wp
from FoamUtils.ThermophysicalProperties import ThermophysicalProperties


def main():
    from data import Ampofo
    from data import DNS
    from functions import time_average
    # Load reference data
    Ampofo = Ampofo.Ampofo()
    dns = DNS.DNS("db_1e10_lin")
    dns_Y0p5 = dns.load_data("Basic_stat/Basic_stat_X_0p5.dat")
    dns_X0p5 = dns.load_data("Data_midwidth/Data_midwidth.dat")
    dns_Nu_cold = dns.load_data("Nu/Nu_cold.dat")
    dns_Nu_hot = dns.load_data("Nu/Nu_hot.dat")

    image_folder = pathlib.Path("Images/temp/")
    image_folder.mkdir(parents=True, exist_ok=True)

    folders = [
        # "postProcessing",
        # "./cases/DNS_Ra1e10/no_buoyancy_source/2025-05-06_Ampofo_laminar_W0.0005_B0.01_BtsFalse_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/no_buoyancy_source/2025-05-06_Ampofo_v2f_W0.0005_B0.01_BtsFalse_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/SGDH/2025-05-06_Ampofo_v2fBuoyant_W0.0005_B0.01_BtsTrue_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/GGDH/2025-05-07_Ampofo_v2fBuoyant_W0.0005_B0.015_BtsTrue_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/GGDH/2025-05-07_Ampofo_v2fBuoyant_W0.0005_B0.01_BtsTrue_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/GGDH/2025-05-07_Ampofo_v2fBuoyant_W0.0005_B0.00667_BtsTrue_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/GGDH/2025-05-06_Ampofo_v2fBuoyant_W0.0005_B0.01_BtsTrue_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/no_buoyancy_source/2025-05-06_Ampofo_kOmegaSST_W0.0005_B0.01_BtsFalse_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/SGDH/2025-05-06_Ampofo_kOmegaSST_W0.0005_B0.01_BtsTrue_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/GGDH/2025-05-06_Ampofo_kOmegaSST_W0.0005_B0.01_BtsTrue_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/no_buoyancy_source/2025-05-06_Ampofo_LaunderSharmaKE_W0.0005_B0.01_BtsFalse_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/SGDH/2025-05-06_Ampofo_LaunderSharmaKE_W0.0005_B0.01_BtsTrue_LTS-False_maxCo0.5/postProcessing/",
        # "./cases/DNS_Ra1e10/GGDH/2025-05-06_Ampofo_LaunderSharmaKE_W0.0005_B0.01_BtsTrue_LTS-False_maxCo0.5/postProcessing/",

        "./cases/DNS_Ra1e10/v2f_GCI/no_buoyancy_source/2025-05-12_Ampofo_v2f_W0.0003_B0.006_BtsFalse_LTS-False_maxCo0.5/postProcessing/",
        "./cases/DNS_Ra1e10/v2f_GCI/no_buoyancy_source/2025-05-12_Ampofo_v2f_W0.0003_B0.009_BtsFalse_LTS-False_maxCo0.5/postProcessing/",
        "./cases/DNS_Ra1e10/v2f_GCI/no_buoyancy_source/2025-05-12_Ampofo_v2f_W0.0003_B0.0135_BtsFalse_LTS-False_maxCo0.5/postProcessing/",

    ]
    # labels = ["laminar"] + ["$v^2-f$ " + s for s in ["", "SGDH", "GGDH"]]
    # labels = ["laminar", "$v^2-f$", "$k-\\omega$ SST", "LS $k-\\epsilon$"]
    # labels = ["$v^2-f$ " + s for s in ["", "SGDH", "GGDH"]]
    labels = [f"{h} mm" for h in ["6", "9", "13.5"]]
    # labels = ["$k-\\omega$ SST " + s for s in ["", "SGDH", "GGDH"]]
    # labels = ["$k-\\epsilon$ LS" + s for s in ["", "SGDH", "GGDH"]]
    names = convertLables(labels)

    # linestyles = ["-", "-"]*4
    # colors = ["gray", "C0", "C1", "C2", "C3"]
    colors = ["k"]*3
    linestyles = ["-.", "--", "-"]
    thermo = ThermophysicalProperties("constant/thermophysicalProperties")

    # Standard figure settings
    figsize = (3.5, 2.5)  # inches
    fontsize = 9
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "lines.linewidth": 0.75,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "figure.dpi": 300,
    })

    df = create_nusselt_dataframe(names + ["DNS"])


    figs, axes = [], []
    for i in range(8):
        f, a = plt.subplots(figsize=figsize)
        figs.append(f)
        axes.append(a)

    for i, postProcessing in enumerate(folders):
        parameters = fu.read_parameters(f"{postProcessing}/../parameters")
        Th, Tc = parameters["T_left"], parameters["T_right"]
        DeltaT = Th - Tc
        T0 = (Th + Tc) / 2
        L = parameters["L_x"]
        Ra = parameters["Ra"]

        alpha = thermo.alpha(1e5, T0)
        kappa = thermo.kappa(T0)
        uNorm = alpha * np.sqrt(Ra) / L
        kNorm = uNorm**2
        tNorm = (Ra**0.5)*kappa/(L**2)
        print(f"{tNorm=}")
        tNorm = (L**2) * (Ra**-0.5)/alpha
        print(f"{tNorm=}")

        label = labels[i]
        name = names[i]
        color = colors[i]
        ls = linestyles[i]
        times = ["600"] # fu.get_sorted_times(f"{postProcessing}/linesample")
        n_times = 1
        for j, time in enumerate(times[-n_times:]):

            alpha = (1/n_times) + j * (1/n_times)

            plot_params = {"label": label, "color": color, "ls": ls, "alpha": alpha}

            vertical = fu.load_sampling_set(f"{postProcessing}/linesample", time, "vertical", org=True)
            horizontal = fu.load_sampling_set(f"{postProcessing}/linesample", time, "horizontal", org=True)
            try:
                right = pd.read_csv(f"{postProcessing}/right_wall/{time}/patchCutLayerAverage.csv")
                left = pd.read_csv(f"{postProcessing}/left_wall/{time}/patchCutLayerAverage.csv")
                print("  Loaded patch data from patchCutLayerAverage.csv")
            except:
                right = fu.load(f"{postProcessing}/right_wall/{time}/line.xy")
                left = fu.load(f"{postProcessing}/left_wall/{time}/line.xy")
                print("  Loaded patch data from line.xy")

            z = vertical["z"]
            x = horizontal["x"]
            T_vert = vertical["TMean"]
            Ux_vert = vertical["UMeanx"]
            Uz_hor = horizontal["UMeanz"]
            T_hor = horizontal["TMean"]
            k_hor = horizontal["kMean"] if parameters["RASModel"] != "laminar" else x * 0

            z_left = left["z"]
            q_left = left["wallHeatFluxMean"]
            q_left_i = left["wallHeatFlux"]
            z_right = right["z"]
            q_right = right["wallHeatFluxMean"]
            q_right_i = right["wallHeatFlux"]
            Nu_left = wp.Nusselt(q_left, DeltaT, L, thermo.kappa(T0))
            Nu_left_i = wp.Nusselt(q_left_i, DeltaT, L, thermo.kappa(T0))
            Nu_right = wp.Nusselt(q_right, DeltaT, L, thermo.kappa(T0))
            Nu_right_i = wp.Nusselt(q_right_i, DeltaT, L, thermo.kappa(T0))

            Q_left_i = np.trapezoid(q_left_i, z_left)*L
            print(f"{Q_left_i=}")

            Nu_mean_left = np.trapezoid(q_left, z_left)/(DeltaT*thermo.kappa(T0))
            Nu_mean_right = -np.trapezoid(q_right, z_right)/(DeltaT*thermo.kappa(T0))

            print(f"  {Nu_mean_left=}")
            print(f"  {Nu_mean_right=}")
            df.loc["Nu_hot", name] = Nu_mean_left
            df.loc["Nu_cold", name] = Nu_mean_right

            # Ux vs z
            fig, ax = figs[0], axes[0]
            ax.plot(Ux_vert, z/L, **plot_params)

            # T vs z
            fig, ax = figs[1], axes[1]
            ax.plot((T_vert - T0)/DeltaT, z/L, **plot_params)

            # Nu_hot vs z
            fig, ax = figs[2], axes[2]
            ax.plot(Nu_left, z_left/L, **plot_params)
            i_params = dict(plot_params, color="r")
            ax.plot(Nu_left_i, z_left/L, **i_params)

            # Nu_cold vs z
            fig, ax = figs[3], axes[3]
            ax.plot(-Nu_right, z_right/L, **plot_params)
            # ax.plot(-Nu_right_i, z_right/L, **plot_params)

            # Uz vs x
            fig, ax = figs[4], axes[4]
            ax.plot(x/L, Uz_hor/uNorm, **plot_params)

            # T vs x
            fig, ax = figs[5], axes[5]
            ax.plot(x/L, (T_hor - T0)/DeltaT, **plot_params)

            # k vs x
            fig, ax = figs[6], axes[6]
            ax.plot(x/L, k_hor/kNorm, **plot_params)


        # Nu vs time
        fig, ax = figs[7], axes[7]

        data = fu.load_all_times(f"{postProcessing}/Q_left", "surfaceFieldValue.dat")
        t, Q_left = data["Time"].to_numpy()/tNorm, data["areaIntegrate(wallHeatFlux)"]
        Nu_left = Q_left / L / ((Th - Tc) * thermo.kappa(T0))
        ax.plot(t, Nu_left, label=f"$Nu_h$ {labels[i]}", color=color)

        t_mask = t*tNorm > 200
        t1, t2 = t[t_mask][0], t[t_mask][-1]
        dt = t2 - t1
        Nu_hot_mean = np.trapezoid(Nu_left[t_mask], t[t_mask])/dt

        print(f"  Nu_hot_mean: {Nu_hot_mean: .2f}")

        data = fu.load_all_times(f"{postProcessing}/Q_right", "surfaceFieldValue.dat")
        t, Q_right = data["Time"].to_numpy()/tNorm, data["areaIntegrate(wallHeatFlux)"]
        Nu_right = -Q_right / L / ((Th - Tc) * thermo.kappa(T0))
        ax.plot(t, Nu_right, label=f"$Nu_c$ {labels[i]}", color=color, ls="--")

        t_mask = t*tNorm > 200
        t1, t2 = t[t_mask][0], t[t_mask][-1]
        dt = t2 - t1
        Nu_cold_mean = np.trapezoid(Nu_right[t_mask], t[t_mask])/dt
        print(f"  Nu_cold_mean: {Nu_cold_mean: .2f}")

    filenames = [
        "Ux", "Tvert", "Nu_hot", "No_cold", "Uhor", "Phi", "k", "Nu" 
    ]

    # Ux vs z
    fig, ax = figs[0], axes[0]
    ax.plot(dns_X0p5["V"], dns_X0p5["x"], **dns.plot_params)
    ax.set_xlabel("$u/u^*$")
    ax.set_ylabel("$y/y^*$")

    # T vs z
    fig, ax = figs[1], axes[1]
    ax.plot(dns_X0p5["T"], dns_X0p5["x"], **dns.plot_params)
    ax.set_xlabel("$\Phi$")
    ax.set_ylabel("$y/y^*$")

    # Nu_hot vs z
    fig, ax = figs[2], axes[2]
    ax.plot(-dns_Nu_hot["Nu_hot"], dns_Nu_hot["x"], **dns.plot_params)
    ax.set_xlabel("Nu$_\\text{hot}$")
    ax.set_ylabel("$y/y^*$")

    dnsNuHot = np.trapezoid(-dns_Nu_hot["Nu_hot"], dns_Nu_hot["x"])
    dnsNuCold = np.trapezoid(-dns_Nu_cold["Nu_cold"], dns_Nu_cold["x"])
    df.loc["Nu_hot", "DNS"] = dnsNuHot
    df.loc["Nu_cold", "DNS"] = dnsNuCold
    df.attrs.update(Ra=1e10)
    df.to_csv(f"{image_folder}/Nu.csv", float_format=".2f")
    print(f"{dnsNuHot=}")
    print(f"{dnsNuCold=}")

    # Nu_cold vs z
    fig, ax = figs[3], axes[3]
    ax.plot(-dns_Nu_cold["Nu_cold"], dns_Nu_cold["x"], **dns.plot_params)
    ax.set_xlabel("Nu$_\\text{cold}$")
    ax.set_ylabel("$y/y^*$")

    # Uz vs x
    fig, ax = figs[4], axes[4]
    ax.plot(dns_Y0p5["y"], dns_Y0p5["U"], **dns.plot_params)
    ax.set_xlabel("$x/x^*$")
    ax.set_ylabel("$v/u^* $")

    # T vs x
    fig, ax = figs[5], axes[5]
    ax.plot(dns_Y0p5["y"], dns_Y0p5["T"], **dns.plot_params)
    ax.set_xlabel("$x/x^*$ ")
    ax.set_ylabel("$\Phi$")

    # k vs x
    fig, ax = figs[6], axes[6]
    ax.plot(dns_Y0p5["y"], dns_Y0p5["k"], **dns.plot_params)
    ax.set_xlabel("$x/x^*$")
    ax.set_ylabel("$k / u^{*2}$")

    # Nu vs time
    fig, ax = figs[7], axes[7]
    ax.set_xlabel("$t/t^*$")
    ax.set_ylabel("Nu")

    for fig, ax, name in zip(figs, axes, filenames):
        ax.grid(False)
        fig.tight_layout()
        for filetype in ["pdf", "svg"]:
            fig.savefig(f"{image_folder}/{name}_no_legend.{filetype}")
        ax.legend(fancybox=False, frameon=False)
        for filetype in ["pdf", "svg"]:
            fig.savefig(f"{image_folder}/{name}.{filetype}")

    plt.show()

def create_nusselt_dataframe(columns, **metadata):
    """
    Create a DataFrame with two rows (Nu_hot, Nu_cold) and the specified columns.
    Optionally attach metadata via keyword arguments.

    Parameters:
    - columns (list of str): Column names (e.g., turbulence models)
    - metadata (kwargs): Optional metadata to attach via df.attrs

    Returns:
    - pd.DataFrame: Initialized DataFrame with metadata
    """
    df = pd.DataFrame(
        {col: [None, None] for col in columns},
        index=["Nu_hot", "Nu_cold"],
    )
    df.index.name = "Quantity"

    df.attrs.update(metadata)
    return df

def convertLables(labelList):

    def rm(string, char):
        return string.replace(char,"")


    convertedLabels = []
    for l in labelList:
        newLabel = l.strip()
        newLabel = rm(newLabel, "$")
        newLabel = rm(newLabel, "^")
        newLabel = rm(newLabel, "\\")
        convertedLabels.append(newLabel)
    return convertedLabels

if __name__ == "__main__":
    main()

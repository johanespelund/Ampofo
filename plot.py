from FoamUtils.ThermophysicalProperties import ThermophysicalProperties
from FoamUtils import wall_profiles as wp
from FoamUtils import file_utils as fu
import os

from rich import print

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import pandas as pd
import pathlib

from plot_utils import create_nusselt_dataframe, convert_labels

plt.style.use(["science", "high-vis"])


def main():
    from data import Ampofo
    from data import DNS

    # Load reference data
    ampofo = Ampofo.Ampofo()
    dns = DNS.DNS("db_1e11_lin")
    dns_Y0p5 = dns.load_data("Basic_stat/Basic_stat_X_0p5.dat")
    dns_X0p5 = dns.load_data("Data_midwidth/Data_midwidth.dat")
    dns_Nu_cold = dns.load_data("Nu/Nu_cold.dat")
    dns_Nu_hot = dns.load_data("Nu/Nu_hot.dat")

    image_folder = pathlib.Path("Images/lowRe-SST-SGDH-comparison/")
    image_folder.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[
            (0, 0, 0),          # black
            (0, 0.4470, 0.7410),  # blue
            (0.9290, 0.6940, 0.1250),  # yellow
            (0.8500, 0.3250, 0.0980),  # orange
            (0.3010, 0.7450, 0.9330),  # light blue
            (0.4660, 0.6740, 0.1880),  # green
            (0.4940, 0.1840, 0.5560),  # purple
            (0.6350, 0.0780, 0.1840),  # burgundy
        ]
    )

    folders = [

        ## Ra = 1e11

        ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-20_Ampofo_kOmegaSST_W0.0025_B0.003_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
         "SST highRe 2.5w 3.0b", "-", "C0"),

        ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-20_Ampofo_kOmegaSST_W0.0025_B0.0045_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
         "SST highRe 2.5w 4.5b", "--", "C0"),

        ("./cases/Boussinesq-deltaT40K/SST-highRe-noOmegaSource/Ra1e11/2026-02-20_Ampofo_kOmegaSST_W0.0025_B0.0045_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
         "SST highRe noOmega 2.5w 4.5b", "--", "C3"),

        ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-20_Ampofo_kOmegaSST_W0.0025_B0.00675_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
         "SST highRe 2.5w 6.75b", ":", "C0"),

        ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-21_Ampofo_kOmegaSST_W0.0015_B0.0045_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
         "SST highRe 1.5w 4.5b", "-", "C4"),

        ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-21_Ampofo_kOmegaSST_W0.0035_B0.0045_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
         "SST highRe 3.5w 4.5b", "-.", "C4"),

        ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-22_Ampofo_kOmegaSST_W0.0035_B0.0045_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
         "SST highRe OmegaNew 3.5w 4.5b", "-.", "C5"),

        ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-22_Ampofo_kOmegaSST_W0.0025_B0.0045_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
         "SST highRe OmegaNew 2.5w 4.5b", "--", "C5"),

    ]

    labels = [l for _, l, _, _ in folders]
    linestyles = [ls for _, _, ls, _ in folders]
    colors = [c for _, _, _, c in folders]
    folders = [f for f, _, _, _ in folders]

    names = convert_labels(labels)

    df = create_nusselt_dataframe(names + ["DNS"])

    figs, axes = [], []
    for _ in range(8):
        f, a = plt.subplots()
        figs.append(f)
        axes.append(a)

    for i, postProcessing in enumerate(folders):
        thermo = ThermophysicalProperties(
            f"{postProcessing}/../constant/thermophysicalProperties")
        parameters = fu.read_parameters(f"{postProcessing}/../parameters")
        Th, Tc = parameters["T_left"], parameters["T_right"]
        DeltaT = Th - Tc
        T0 = (Th + Tc) / 2
        L = parameters["L_x"]
        Ra = parameters["Ra"]

        kappa = 1004.675 * 1.8e-05 / 0.71
        alpha = kappa / (thermo.rho(parameters["p_outlet"], T0) * 1004.675)
        uNorm = alpha * np.sqrt(Ra) / L
        kNorm = uNorm**2
        tNorm = (L**2) * (Ra**-0.5) / alpha

        label = labels[i]
        name = names[i]
        color = colors[i]
        ls = linestyles[i]
        times = fu.get_sorted_times(f"{postProcessing}/linesample")
        times = [times[-1]]
        print(f"Loading data from {postProcessing} @ t={times[-1]}")

        beta = thermo.beta(parameters["p_outlet"], T0)
        print(f"  {beta=}")
        print(f"  {1/T0=}")
        nu = 1.8e-5 / thermo.rho(parameters["p_outlet"], T0)
        print(f"  {nu=}")
        print(f"  {DeltaT=}")
        print(f"  {L=}")
        print(f"  Pr=0.71")
        Ra = 0.71 * 9.81 * beta * DeltaT * L**3 / (nu**2)
        print(f"    Calculated Ra: {Ra:.4e}")

        for j, time in enumerate(times[-1:]):
            mean = "TMean" in os.listdir(f"{postProcessing}/../processor0/{time}/")
            meanLabel = "Mean" if mean else ""

            alpha_plot = (1 / len(times)) + j * (1 / len(times))
            plot_params = {"label": label, "color": color, "ls": ls, "alpha": alpha_plot}

            vertical = fu.load_sampling_set(
                f"{postProcessing}/linesample", time, "vertical", org=True)
            horizontal = fu.load_sampling_set(
                f"{postProcessing}/linesample", time, "horizontal", org=True)
            try:
                right = pd.read_csv(
                    f"{postProcessing}/right_wall/{time}/patchCutLayerAverage.csv")
                left = pd.read_csv(
                    f"{postProcessing}/left_wall/{time}/patchCutLayerAverage.csv")
                print("  Loaded patch data from patchCutLayerAverage.csv")
            except (FileNotFoundError, OSError):
                right = fu.load(f"{postProcessing}/right_wall/{time}/line.xy")
                left = fu.load(f"{postProcessing}/left_wall/{time}/line.xy")
                print("  Loaded patch data from line.xy")

            yPlus = fu.load_all_times(f"{postProcessing}/yPlus/", "yPlus.dat")
            print(f"  max y+ is {yPlus['max'].iloc[-1]}")
            print(f"  average y+ is {yPlus['average'].iloc[-1]}")

            z = vertical["z"]
            x = horizontal["x"]
            T_vert = vertical["T" + meanLabel]
            Ux_vert = vertical["U" + meanLabel + "x"]
            Uz_hor = horizontal["U" + meanLabel + "z"]
            T_hor = horizontal["T" + meanLabel]
            k_hor = horizontal["k" + meanLabel] if parameters["RASModel"] != "laminar" else x * 0

            z_left = left["z"]
            q_left = left["wallHeatFlux" + meanLabel]
            q_left_i = left["wallHeatFlux"]
            z_right = right["z"]
            q_right = right["wallHeatFlux" + meanLabel]
            Nu_left = q_left * L / (DeltaT * kappa)
            Nu_right = wp.Nusselt(q_right, DeltaT, L, kappa)

            Nu_mean_left = np.trapezoid(q_left, z_left) / (DeltaT * kappa)
            Nu_mean_right = -np.trapezoid(q_right, z_right) / (DeltaT * kappa)
            df.loc["Nu_hot", name] = Nu_mean_left
            df.loc["Nu_cold", name] = Nu_mean_right

            # Ux vs z
            axes[0].plot(Ux_vert / uNorm, z / L, **plot_params)

            # T vs z
            axes[1].plot((T_vert - T0) / DeltaT, z / L, **plot_params)

            # Nu_hot vs z
            axes[2].plot(Nu_left, z_left / L, **plot_params)

            # Nu_cold vs z
            axes[3].plot(-Nu_right, z_right / L, **plot_params)

            # Uz vs x
            axes[4].plot(x / L, Uz_hor / uNorm, **plot_params)

            # T vs x
            axes[5].plot(x / L, (T_hor - T0) / DeltaT, **plot_params)

            # k vs x
            axes[6].plot(x[2:-2] / L, k_hor[2:-2] / kNorm, **plot_params)

        # Nu vs time
        fig_nu, ax_nu = figs[7], axes[7]
        data = fu.load_all_times(f"{postProcessing}/Q_left", "surfaceFieldValue.dat")
        t = data["Time"].to_numpy() / tNorm
        Q_left = data["areaIntegrate(wallHeatFlux)"]
        Nu_left_t = Q_left / L / ((Th - Tc) * kappa)
        mask = t > 1
        ax_nu.plot(t[mask], Nu_left_t[mask], label=f"$Nu_h$ {labels[i]}", color=color, ls=ls)

        print(f"   {L=:.3f}")
        print(f"   {Th=:.2f}")
        print(f"   {Tc=:.2f}")
        print(f"   {kappa=:.5f}")

        t_mask = t > t[-1] - 50 * tNorm
        t1, t2 = t[t_mask][0], t[t_mask][-1]
        dt = t2 - t1
        Nu_hot_mean = np.trapezoid(Nu_left_t[t_mask], t[t_mask]) / dt
        print(f"  Nu_hot_mean: {Nu_hot_mean: .2f}")

        data = fu.load_all_times(f"{postProcessing}/Q_right", "surfaceFieldValue.dat")
        t = data["Time"].to_numpy() / tNorm
        Q_right = data["areaIntegrate(wallHeatFlux)"]
        Nu_right_t = -Q_right / L / ((Th - Tc) * kappa)

        t_mask = t > t[-1] - 50 * tNorm
        t1, t2 = t[t_mask][0], t[t_mask][-1]
        dt = t2 - t1
        Nu_cold_mean = np.trapezoid(Nu_right_t[t_mask], t[t_mask]) / dt
        print(f"  Nu_cold_mean: {Nu_cold_mean: .2f}")

    filenames = ["Ux", "Tvert", "Nu_hot", "Nu_cold", "Uhor", "Phi", "k", "Nu"]

    # Ux vs z
    axes[0].plot(dns_X0p5["V"], dns_X0p5["x"], **dns.plot_params)
    axes[0].set_xlabel("$u/u^*$")
    axes[0].set_ylabel("$y/L$")

    # T vs z
    axes[1].plot(dns_X0p5["T"], dns_X0p5["x"], **dns.plot_params)
    axes[1].set_xlabel(r"$\Phi$")
    axes[1].set_ylabel("$y/L$")

    # Nu_hot vs z
    axes[2].plot(-dns_Nu_hot["Nu_hot"], dns_Nu_hot["x"], **dns.plot_params)
    axes[2].set_xlabel("Nu$_\\text{hot}$")
    axes[2].set_ylabel("$y/L$")

    dnsNuHot = np.trapezoid(-dns_Nu_hot["Nu_hot"], dns_Nu_hot["x"])
    dnsNuCold = np.trapezoid(-dns_Nu_cold["Nu_cold"], dns_Nu_cold["x"])
    df.loc["Nu_hot", "DNS"] = dnsNuHot
    df.loc["Nu_cold", "DNS"] = dnsNuCold
    df.attrs.update(Ra=1e11)
    df.to_csv(f"{image_folder}/Nu.csv", float_format=".2f")
    print(f"{dnsNuHot=}")
    print(f"{dnsNuCold=}")

    # Nu_cold vs z
    axes[3].plot(-dns_Nu_cold["Nu_cold"], dns_Nu_cold["x"], **dns.plot_params)
    axes[3].set_xlabel("Nu$_\\text{cold}$")
    axes[3].set_ylabel("$y/L$")

    # Uz vs x
    axes[4].plot(dns_Y0p5["y"], dns_Y0p5["U"], **dns.plot_params)
    axes[4].set_xlabel("$x/x^*$")
    axes[4].set_ylabel("$v/u^* $")

    # T vs x
    axes[5].plot(dns_Y0p5["y"], dns_Y0p5["T"], **dns.plot_params)
    axes[5].set_xlabel("$x/x^*$ ")
    axes[5].set_ylabel(r"$\Phi$")

    # k vs x
    axes[6].plot(dns_Y0p5["y"], dns_Y0p5["k"], **dns.plot_params)
    axes[6].set_xlabel("$x/x^*$")
    axes[6].set_ylabel("$k / u^{*2}$")

    # Nu vs time
    axes[7].set_xlabel("$t/t^*$")
    axes[7].set_ylabel("Nu")

    for fig, ax, name in zip(figs, axes, filenames):
        ax.grid(False)
        fig.tight_layout()
        for filetype in ["pdf", "svg"]:
            fig.savefig(f"{image_folder}/{name}_no_legend.{filetype}")
        ax.legend(fancybox=False, frameon=False)
        for filetype in ["pdf", "svg"]:
            fig.savefig(f"{image_folder}/{name}.{filetype}")

    plt.show()


if __name__ == "__main__":
    main()

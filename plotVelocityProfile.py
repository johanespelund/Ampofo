from FoamUtils.ThermophysicalProperties import ThermophysicalProperties
from FoamUtils import wall_profiles as wp
from FoamUtils import file_utils as fu
import os

import subprocess
from rich import print

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy as sp
import pandas as pd
import pathlib
# import pyvista

plt.style.use(["science", "high-vis"])
# # Standard figure settings
# figsize = (3.5, 2.5)  # inches
# fontsize = 9
# plt.rcParams.update({
#     "font.size": fontsize,
#     "axes.labelsize": fontsize,
#     "axes.titlesize": fontsize,
#     "lines.linewidth": 0.75,
#     "legend.fontsize": fontsize,
#     "xtick.labelsize": fontsize,
#     "ytick.labelsize": fontsize,
#     "figure.dpi": 300,
# })


mean = False
meanLabel = "Mean" if mean else ""


def main():
    from data import Ampofo
    from data import DNS
    from functions import time_average
    # Load reference data
    Ampofo = Ampofo.Ampofo()
    # dns = DNS.DNS("db_1e10_lin")
    dns = DNS.DNS("db_1e11_lin")
    dns_Y0p5 = dns.load_data("Basic_stat/Basic_stat_X_0p5.dat")
    dns_X0p5 = dns.load_data("Data_midwidth/Data_midwidth.dat")
    dns_Nu_cold = dns.load_data("Nu/Nu_cold.dat")
    dns_Nu_hot = dns.load_data("Nu/Nu_hot.dat")

    image_folder = pathlib.Path("Images/lowRe-SST-SGDH-comparison/")
    image_folder.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[

            (0, 0, 0),  # black
            (0, 0.4470, 0.7410),  # blue
            (0.9290, 0.6940, 0.1250),  # yellow
            (0.8500, 0.3250, 0.0980),  # orange
            (0.3010, 0.7450, 0.9330),  # light blue
            (0.4660, 0.6740, 0.1880),  # green
            (0.4940, 0.1840, 0.5560),  # purple
            (0.6350, 0.0780, 0.1840),  # burgundy

        ]
    )
    P1 = pathlib.Path("./cases/perfectGas/")
    folders = [

        ## Ra = 1e10

        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e10/2026-01-28_Ampofo_kOmegaSST_W0.0025_B0.0045_BtsFalse-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "SST highRe", "-", "C0"),
        #
        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e10/2026-01-28_Ampofo_kOmegaSST_W0.0025_B0.0045_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "SST highRe + Pb", "--", "C0"),

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

        # ("./cases/Boussinesq-deltaT40K/v2fBuoyant-noTANH/Ra1e11/2026-02-20_Ampofo_v2fBuoyant_W0.0005_B0.004_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "v2f lowRe noTANH 0.5w 4.0b", "--", "C1"),
        #
        # ("./cases/Boussinesq-deltaT40K/v2fBuoyant/Ra1e11/2026-02-20_Ampofo_v2fBuoyant_W0.0005_B0.004_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "v2f lowRe TANH 0.5w 4.0b", "-", "C1"),

        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-15_Ampofo_v2fBuoyant_W0.00025_B0.004_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 0.25 \\cdot 10^{-3}$", "-", "C1"),
        #
        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-14_Ampofo_v2fBuoyant_W0.0005_B0.004_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 0.5 \\cdot 10^{-3}$", "--", "C1"),
        #
        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-14_Ampofo_v2fBuoyant_W0.001_B0.004_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 1.0 \\cdot 10^{-3}$", "-.", "C1"),
        #
        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-14_Ampofo_v2fBuoyant_W0.002_B0.004_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 2.0 \\cdot 10^{-3}$", ":", "C1"),

        # ("./cases/perfectGas-deltaT40K/SST-highRe/Ra1e11/2026-02-16_Ampofo_v2fBuoyant_W0.0005_B0.004_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 0.5 \\cdot 10^{-3}$ IG", "--", "C0"),
        #
        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-16_Ampofo_kOmegaSST_W0.005_B0.0075_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 5 \\cdot 10^{-3}$ SST", ":", "C2"),
        #
        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-17_Ampofo_kOmegaSST_W0.0025_B0.005_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 2.5 \\cdot 10^{-3}$ SST", "--", "C2"),
        #
        #
        # ("./cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-16_Ampofo_kOmegaSST_W0.00125_B0.005_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 1.25 \\cdot 10^{-3}$ SST", "-", "C2"),
        #
        # ("./cases/Boussinesq-deltaT40K-kLowReWF/SST-highRe/Ra1e11/2026-02-17_Ampofo_kOmegaSST_W0.00125_B0.005_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 1.25 \\cdot 10^{-3}$ SST kLowReWF", "-", "C3"),
        #
        # ("./cases/Boussinesq-deltaT40K-kqR-nutk/SST-highRe/Ra1e11/2026-02-17_Ampofo_kOmegaSST_W0.005_B0.005_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 5 \\cdot 10^{-3}$ SST kqR nutk", "-", "C4"),



        # ("./cases/Boussinesq-deltaT40K-kLowReWF/SST-highRe/Ra1e11/2026-02-17_Ampofo_kOmegaSST_W0.01_B0.01_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 10 \\cdot 10^{-3}$ SST kLowReWF", "--", "C3"),
        #
        # ("./cases/Boussinesq-deltaT40K-kqR-nutk/SST-highRe/Ra1e11/2026-02-17_Ampofo_kOmegaSST_W0.01_B0.01_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 10 \\cdot 10^{-3}$ SST kqR nutk", "--", "C4"),
        #
        #
        # ("./cases/Boussinesq-deltaT40K-kLowReWF/SST-highRe/Ra1e11/2026-02-17_Ampofo_kOmegaSST_W0.02_B0.02_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 20 \\cdot 10^{-3}$ SST kLowReWF", "-.", "C3"),
        #
        #
        # ("./cases/Boussinesq-deltaT40K-kqR-nutk/SST-highRe/Ra1e11/2026-02-17_Ampofo_kOmegaSST_W0.02_B0.02_BtsTrue-SGDH_Prt-0.85_maxCo0.5/postProcessing/",
        #  "$\Delta x_w = 20 \\cdot 10^{-3}$ SST kqR nutk", "-.", "C4"),


    ]
    # labels = ["laminar"] + ["$v^2-f$ " + s for s in ["", "SGDH", "GGDH"]]
    # labels = ["laminar", "$k-\\omega$ SST", "$v^2-f$", "LS $k-\\epsilon$"]
    # labels = ["laminar","$k-\\omega$ SST", "RNG $k-\\epsilon$", "realizable $k-\\epsilon$"]
    # labels = ["$v^2-f$ " + s + " (6 mm)" for s in ["", "SGDH", "GGDH"]]
    # labels =  [f"$\phi_t-f$ + Pb (SGDH) {h} mm" for h in [6]] + ["no Pb"] + ["ref."]
    labels = [l for _, l, _, _ in folders]
    linestyles = [ls for _, _, ls, _ in folders]
    colors = [c for _, _, _, c in folders]
    folders = [f for f, _, _, _ in folders]

    # labels = [f"SST Pb", "SST"]
    # labels =  [f"{h} mm" for h in [5, 7.5, 11.25]]
    # labels = [f"{h} mm" for h in ["4" ,"6", "9", "13.5"]]
    # labels = [f"{h} mm" for h in ["6", "9", "13.5"]]
    # labels = ["orthogonal", "corrected"]
    # labels = ["$k-\\omega$ SST " + s for s in ["", "SGDH", "GGDH"]]
    # labels = ["$k-\\epsilon$ LS" + s for s in ["", "SGDH", "GGDH"]]
    names = convertLables(labels)

    # linestyles = ["-", "-"]*4
    # colors = ["C0", "C1"]*3 #, "C2", "C3"]*2  # , "C4", "C5"]
    # colors = ["k"]*4
    # linestyles = ["-", "--", "-."]
    # linestyles = ["-"]*4 + ["--"]*4

    df = create_nusselt_dataframe(names + ["DNS"])

    figs, axes = [], []
    for i in range(8):
        f, a = plt.subplots()  # figsize=figsize)
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

        # if thermo.properties["thermoType"]["equationOfState"] == "Boussinesq":
        #     kappa = 1.4*717 * 1.8e-5 / 0.71
        # else:
        kappa = 1004.675 * 1.8e-05 / 0.71  # thermo.kappa(T0)

        # thermo.alpha(1e5, T0)
        alpha = kappa / (thermo.rho(parameters["p_outlet"], T0) * 1004.675)
        uNorm = alpha * np.sqrt(Ra) / L
        kNorm = uNorm**2
        # tNorm = (Ra**0.5)*kappa/(L**2)
        # print(f"{tNorm=}")
        tNorm = (L**2) * (Ra**-0.5)/alpha

        label = labels[i]
        name = names[i]
        color = colors[i]
        ls = linestyles[i]
        times = fu.get_sorted_times(f"{postProcessing}/linesample")
        times = [times[-1]]
        # times = ["50"] #[times[-1]]
        print(f"Loading data from {postProcessing} @ t={times[-1]}")
        n_times = 1

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

        for j, time in enumerate(times[-n_times:]):

            

            mean = "TMean" in os.listdir(f"{postProcessing}/../processor0/{time}/")
            # mean = False
            meanLabel = "Mean" if mean else ""

            # open(f"{postProcessing}/../open.foam", "a").close()
            # reader = pyvista.POpenFOAMReader(f"{postProcessing}/../open.foam")
            # reader.case_type = "decomposed"
            # _times = reader.time_values
            # reader.set_active_time_value(_times[-1])
            # reader.cell_to_point_creation = True
            # mesh = reader.read()
            # # internalMesh = mesh['internalMesh']
            # # sample = internalMesh.sample_over_line((0, 0.1*L/2, 0.995*L/2), (L, 0.1*L/2, 0.995*L/2)) #, resolution=1000)
            # # print(f"{sample=}")

            # if "Pb" in mesh["internalMesh"].cell_data:
            #     plotter = pyvista.Plotter()
            #     plotter.add_mesh(mesh["internalMesh"], scalars="Pb")
            #     plotter.view_xz()
            #     plotter.enable_parallel_projection()
            #     plotter.show()

            alpha = (1/n_times) + j * (1/n_times)

            plot_params = {"label": label,
                           "color": color, "ls": ls, "alpha": alpha}

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
            except:
                right = fu.load(f"{postProcessing}/right_wall/{time}/line.xy")
                left = fu.load(f"{postProcessing}/left_wall/{time}/line.xy")
                print("  Loaded patch data from line.xy")

            yPlus = fu.load_all_times(f"{postProcessing}/yPlus/", "yPlus.dat")
            print(f"  max y+ is {yPlus['max'].iloc[-1]}")
            print(f"  average y+ is {yPlus['average'].iloc[-1]}")


            # for df in [right, left]:
            #     df.columns = df.columns.str.replace(
            #         'M...', 'Mean', regex=False)

            z = vertical["z"]
            x = horizontal["x"]
            T_vert = vertical["T" + meanLabel]
            Ux_vert = vertical["U" + meanLabel + "x"]
            Uz_hor = horizontal["U" + meanLabel + "z"]
            T_hor = horizontal["T" + meanLabel]
            k_hor = horizontal["k" +
                               meanLabel] if parameters["RASModel"] != "laminar" else x * 0

            z_left = left["z"]
            q_left = left["wallHeatFlux" + meanLabel]
            q_left_i = left["wallHeatFlux"]
            z_right = right["z"]
            q_right = right["wallHeatFlux" + meanLabel]
            q_right_i = right["wallHeatFlux"]
            # wp.Nusselt(q_left, DeltaT, L, kappa)
            Nu_left = q_left * L / (DeltaT * kappa)
            Nu_left_i = wp.Nusselt(q_left_i, DeltaT, L, kappa)
            Nu_right = wp.Nusselt(q_right, DeltaT, L, kappa)
            Nu_right_i = wp.Nusselt(q_right_i, DeltaT, L, kappa)

            Q_left_i = np.trapezoid(q_left_i, z_left)*L

            Nu_mean_left = np.trapezoid(q_left, z_left)/(DeltaT*kappa)
            Nu_mean_right = -np.trapezoid(q_right, z_right)/(DeltaT*kappa)

            # print(f"  {Nu_mean_left=}")
            # print(f"  {Nu_mean_right=}")
            df.loc["Nu_hot", name] = Nu_mean_left
            df.loc["Nu_cold", name] = Nu_mean_right

            # Ux vs z
            fig, ax = figs[0], axes[0]
            ax.plot(Ux_vert/uNorm, z/L, **plot_params)

            # T vs z
            fig, ax = figs[1], axes[1]
            ax.plot((T_vert - T0)/DeltaT, z/L, **plot_params)

            # Nu_hot vs z
            fig, ax = figs[2], axes[2]
            ax.plot(Nu_left, z_left/L, **plot_params)
            # i_params = dict(plot_params, color="r")
            # ax.plot(Nu_left_i, z_left/L, **i_params)

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
            # ax.plot(x[2:-2]/L, k_hor[2:-2]/kNorm, **(plot_params|{"marker": "x"}))
            ax.plot(x[2:-2]/L, k_hor[2:-2]/kNorm, **plot_params)
            # ax.plot(sample["Distance"]/L, sample.get_array("kMean")/kNorm, **plot_params)

        # Nu vs time
        t_start = 0
        if True:
            fig, ax = figs[7], axes[7]

            data = fu.load_all_times(
                f"{postProcessing}/Q_left", "surfaceFieldValue.dat")
            t, Q_left = data["Time"].to_numpy(
            )/tNorm, data["areaIntegrate(wallHeatFlux)"]
            Nu_left = Q_left / L / ((Th - Tc) * kappa)
            mask = t > 1
            ax.plot(
                t[mask], Nu_left[mask], label=f"$Nu_h$ {labels[i]}", color=color, ls=ls)
            # ax.plot(t, Q_left, label=f"$Q_h$ {labels[i]}", color=color)
            print(f"   {L=:.3f}")
            print(f"   {Th=:.2f}")
            print(f"   {Tc=:.2f}")
            print(f"   {kappa=:.5f}")

            # print(f"{Q_left=}")

            # t_mask = t*tNorm > t_start
            t_mask = t > t[-1] - 50*tNorm
            t1, t2 = t[t_mask][0], t[t_mask][-1]
            dt = t2 - t1
            Nu_hot_mean = np.trapezoid(Nu_left[t_mask], t[t_mask])/dt

            print(f"  Nu_hot_mean: {Nu_hot_mean: .2f}")

            data = fu.load_all_times(
                f"{postProcessing}/Q_right", "surfaceFieldValue.dat")
            t, Q_right = data["Time"].to_numpy(
            )/tNorm, data["areaIntegrate(wallHeatFlux)"]
            mask = t > 1
            Nu_right = -Q_right / L / ((Th - Tc) * kappa)
            # ax.plot(t[mask], Nu_right[mask], label=f"$Nu_c$ {labels[i]}", color=color, ls="--", ls=ls)

            # t_mask = t*tNorm > t_start
            t_mask = t > t[-1] - 50*tNorm
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
    ax.set_ylabel("$y/L$")

    # T vs z
    fig, ax = figs[1], axes[1]
    ax.plot(dns_X0p5["T"], dns_X0p5["x"], **dns.plot_params)
    ax.set_xlabel("$\Phi$")
    ax.set_ylabel("$y/L$")

    # Nu_hot vs z
    fig, ax = figs[2], axes[2]
    ax.plot(-dns_Nu_hot["Nu_hot"], dns_Nu_hot["x"], **dns.plot_params)
    ax.set_xlabel("Nu$_\\text{hot}$")
    ax.set_ylabel("$y/L$")

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
    ax.set_ylabel("$y/L$")

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
        return string.replace(char, "")

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

import os
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy as sp

plt.style.use(["science", "nature", "grid"])

from FoamUtils import file_utils as fu
from FoamUtils import wall_profiles as wp
from FoamUtils.ThermophysicalProperties import ThermophysicalProperties

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

folders = ["postProcessing"]
labels = ["$v^2-f$"]
linestyles = ["-"]
colors = ["C0"]
thermo = ThermophysicalProperties("constant/thermophysicalProperties")

# Standard figure settings
figsize = (3.5, 2.5)  # inches
fontsize = 9
plt.rcParams.update({
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "figure.dpi": 300,
})

for i, postProcessing in enumerate(folders):
    parameters = fu.read_parameters(f"{postProcessing}/../parameters")
    Th, Tc = parameters["T_left"], parameters["T_right"]
    DeltaT = Th - Tc
    T0 = (Th + Tc) / 2
    L = parameters["L_x"]
    Ra = parameters["Ra"]

    alpha = thermo.alpha(1e5, T0)
    uNorm = alpha * np.sqrt(Ra) / L
    kNorm = uNorm**2
    tNorm = (L**2) * (Ra**-0.5)/alpha

    label = labels[i]
    color = colors[i]
    ls = linestyles[i]
    time = fu.get_sorted_times(f"{postProcessing}/linesample")[-1]

    vertical = fu.load_sampling_set(f"{postProcessing}/linesample", time, "vertical", org=True)
    horizontal = fu.load_sampling_set(f"{postProcessing}/linesample", time, "horizontal", org=True)
    right = fu.load(f"{postProcessing}/right_wall/{time}/line.xy")
    left = fu.load(f"{postProcessing}/left_wall/{time}/line.xy")

    z = vertical["z"]
    x = horizontal["x"]
    T_vert = vertical["T"]
    Ux_vert = vertical["Ux"]
    Uz_hor = horizontal["UMeanz"]
    T_hor = horizontal["TMean"]
    k_hor = horizontal["kMean"] if parameters["RASModel"] != "laminar" else x * 0

    z_left = left["z"]
    q_left = left["wallHeatFluxM..."]
    z_right = right["z"]
    q_right = right["wallHeatFluxM..."]
    Nu_left = wp.Nusselt(q_left, DeltaT, L, thermo.kappa(Th))
    Nu_right = wp.Nusselt(q_right, DeltaT, L, thermo.kappa(Th))

    # Ux vs z
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(Ux_vert, z/L, label=label, color=color, ls=ls)
    ax.plot(dns_X0p5["V"], dns_X0p5["x"], **dns.plot_params)
    ax.set_xlabel("$u/u^*$")
    ax.set_ylabel("$y/y^*$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{label}_Ux_vs_z.pdf")

    # T vs z
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot((T_vert - T0)/DeltaT, z/L, label=label, color=color, ls=ls)
    ax.plot(dns_X0p5["T"], dns_X0p5["x"], **dns.plot_params)
    ax.set_xlabel("$\Phi$")
    ax.set_ylabel("$y/y*$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{label}_T_vs_z.pdf")

    # Nu_hot vs z
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(Nu_left, z_left/L, label=label, color=color, ls=ls)
    ax.plot(-dns_Nu_hot["Nu_hot"], dns_Nu_hot["x"], **dns.plot_params)
    ax.set_xlabel("Nu$_\\text{hot}$")
    ax.set_ylabel("$y/y^*$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{label}_Nu_vs_z.pdf")

    # Nu_cold vs z
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(Nu_right, 1 - z_right/L, label=label, color=color, ls=ls)
    ax.plot(-dns_Nu_cold["Nu_cold"], dns_Nu_cold["x"], **dns.plot_params)
    ax.set_xlabel("Nu$_\\text{cold}$")
    ax.set_ylabel("$y/y^*$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{label}_NuCold_vs_z.pdf")
    # Uz vs x
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x/L, Uz_hor/uNorm, label=label, color=color, ls=ls)
    ax.plot(dns_Y0p5["y"], dns_Y0p5["U"], **dns.plot_params)
    ax.set_xlabel("$x/x^*$")
    ax.set_ylabel("$v/u^* $")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{label}_Uz_vs_x.pdf")

    # T vs x
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x/L, (T_hor - T0)/DeltaT, label=label, color=color, ls=ls)
    ax.plot(dns_Y0p5["y"], dns_Y0p5["T"], **dns.plot_params)
    ax.set_xlabel("$x/x^*$ ")
    ax.set_ylabel("$\Phi$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{label}_T_vs_x.pdf")

    # k vs x
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x/L, k_hor/kNorm, label=label, color=color, ls=ls)
    ax.plot(dns_Y0p5["y"], dns_Y0p5["k"], **dns.plot_params)
    ax.set_xlabel("$x/x^*$")
    ax.set_ylabel("$k / u^{*2}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{label}_k_vs_x.pdf")


    # Nu vs time
    fig, ax = plt.subplots(figsize=figsize)

    data = fu.load_all_times(f"{postProcessing}/Q_left", "surfaceFieldValue.dat")
    t, Q_left = data["Time"].to_numpy()/tNorm, data["areaIntegrate(wallHeatFlux)"]
    Nu_left = Q_left / L / ((Th - Tc) * thermo.kappa(T0))
    ax.plot(t, Nu_left, label=f"$Nu_h$ {labels[i]}", color=color)

    t_mask = t > 60
    t1, t2 = t[t_mask][0], t[t_mask][-1]
    dt = t2 - t1
    Nu_hot_mean = np.trapezoid(Nu_left[t_mask], t[t_mask])/dt

    print(f"Nu_hot_mean: {Nu_hot_mean: .2f}")

    data = fu.load_all_times(f"{postProcessing}/Q_right", "surfaceFieldValue.dat")
    t, Q_right = data["Time"].to_numpy()/tNorm, data["areaIntegrate(wallHeatFlux)"]
    Nu_right = -Q_right / L / ((Th - Tc) * thermo.kappa(T0))
    ax.plot(t, Nu_right, label=f"$Nu_c$ {labels[i]}", color=color, ls="--")

    t_mask = t > 60
    t1, t2 = t[t_mask][0], t[t_mask][-1]
    dt = t2 - t1
    Nu_cold_mean = np.trapezoid(Nu_right[t_mask], t[t_mask])/dt
    print(f"Nu_cold_mean: {Nu_cold_mean: .2f}")




ax.set_xlabel("$t/t^*$")
ax.set_ylabel("Nu")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig("Nu_vs_t.pdf")
plt.show()

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scienceplots

plt.style.use(["science", "nature", "grid"])

from FoamUtils import file_utils as fu
from FoamUtils import wall_profiles as wp
from FoamUtils.ThermophysicalProperties import ThermophysicalProperties

from data.Tian import Tian as exp
from data.Ampofo import Ampofo
exp = exp()
Ampofo = Ampofo()

from functions import time_average

folders = [
    "postProcessing",
    "./cases/kOmegaSST/1cm0.3mm_noSource/postProcessing",
    "./cases/kOmegaSST/2cm0.3mm_noSource/postProcessing",
    # "./cases/kOmegaSST_noSource2cm0.3mm/postProcessing",
    # "./cases/kOmegaSST_noSource1cm0.3mm/postProcessing",
    # "./cases/kOmegaSST_noSource0.5cm0.3mm/postProcessing",
]

thermo = ThermophysicalProperties("constant/thermophysicalProperties")

# Read the n last velocity profiles
parameters = fu.read_parameters("parameters")
Th, Tc = parameters["T_left"], parameters["T_right"]
T0 = (Th + Tc) / 2
L = parameters["L_x"]
V0 = np.sqrt(9.81*L*(Th-Tc)/T0)
print(thermo.kappa(T0))

j = 0
fig, ax = plt.subplots(2, 3, figsize=(9, 4))
ax = ax.flatten()

fig, ax2 = plt.subplots(1, 3, figsize=(7, 3))
ax[2] = ax2[0]
ax[3] = ax2[1]
ax[4] = ax2[2]


k = 0
for postProcessing in folders:
    label = postProcessing.split("/")[0]
    sorted_folders = fu.get_sorted_times(f"{postProcessing}/linesample")
    time = sorted_folders[-1]
    # sorted_folders = [time]
    sorted_folders = sorted_folders[-1:]
    # sorted_folders = ["160", "300"]
    pp_color = f"C{k}"
    k += 1

    for i, folder in enumerate(sorted_folders):
        vertical = fu.load_sampling_set(
            f"{postProcessing}/linesample", folder, "vertical"
        )
        horizontal = fu.load_sampling_set(
            f"{postProcessing}/linesample", folder, "horizontal"
        )
        right = fu.load_sampling_set(
            f"{postProcessing}/wallsample", folder, "rightwall"
        )
        left = fu.load_sampling_set(
            f"{postProcessing}/wallsample", folder, "leftwall"
        )

        z = vertical["z"] #vertical[:, 0]
        z_right = right["z"]
        z_left = left["z"]
        T_vert = vertical["T"] #[:, iT]
        Ux_vert = vertical["UMeanx"] #[:, iUx]
        Uz_vert = vertical["UMeanx"] #:, iUz]


        q_right = right["wallHeatFluxMean"] #[:, 1]
        q_left = left["wallHeatFluxMean"] #[:, 1

        Nu_left = wp.Nusselt(q_left, Th - Tc, L, thermo.kappa(Th))
        Nu_right = wp.Nusselt(q_right, Th - Tc, L, thermo.kappa(Tc))


        T_hor = horizontal["TMean"] #[:, 1]
        Uz_hor = horizontal["UMeanz"] #[:, iUz]
        x = horizontal["x"] #[:, 0]

        Uz_left = horizontal["UMeanz"] #[:, iUz]
        Uz_right = np.flip(Uz_left)

        rho_data = horizontal["rhoMean"]

        n_left = horizontal["x"]
        n_right = L - np.flip(n_left)
        rho_left = rho_data[0]
        # rho_right = rho_data[-1]
        rho_right = rho_data.to_numpy()[-1]

        T_left = T_hor
        T_right = np.flip(T_hor)

        uPlus_left = wp.u_plus(Uz_left, T_left, n_left, rho_left, thermo)
        uPlus_right = wp.u_plus(Uz_right, T_right, n_right, rho_right, thermo)

        nPlus_left = wp.n_plus(Uz_left, T_left, n_left, rho_left, thermo)
        nPlus_right = wp.n_plus(Uz_right, T_right, n_right, rho_right, thermo)
        
        Tref = 0.5 * (Th + Tc)
        phiPlus_left = wp.phi_plus(Uz_left, T_left, n_left, rho_left, Tref, thermo)
        phiPlus_right = wp.phi_plus(Uz_left, T_right, n_right, rho_right, Tref, thermo)

        ax[0].plot(Ux_vert, z, label=label, marker="x", markevery=2, color=pp_color)
        ax[0].set_ylabel("z/L [-]")
        ax[0].set_xlabel("Ux [m/s]")

        ax[1].plot(T_vert, z, label=label, marker="x", markevery=2, color=pp_color)
        ax[1].set_ylabel("z/L [-]")
        ax[1].set_xlabel("T [K]")

        # # Nu_h = np.trapezoid(Nu_left, z)/L
        # print(f"{label} Nu_h = {Nu_h}")
        Nu_left = q_left * L / ((Th - Tc) * thermo.kappa(Th))
        ax[2].plot(
            Nu_left,
            # q_left,
            z_left/L,
            label="Left wall",
            # marker="o",
            # markevery=2,
            color=pp_color,
            markersize=1,
        )
        ax[2].plot(Ampofo.Nu_hot[:,1], Ampofo.Nu_hot[:,0], "x", label=Ampofo.label+" left", color=pp_color)
        # Nu_c = np.trapezoid(Nu_right, z_right)/L
        # print(f"{label} Nu_c = {Nu_c}")
        ax[2].plot(
            -Nu_right,
            # -q_right,
            z_right/L,
            label="Right wall",
            # marker="z",
            ls="--",
            # markevery=2,
            color=pp_color,
            markersize=1,
        )
        ax[2].plot(Ampofo.Nu_cold[:,1], Ampofo.Nu_cold[:,0], "o", label=Ampofo.label+" right", color=pp_color)
    ax[2].set_ylabel("z/L [-]")
    ax[2].set_xlabel("Nu [-]")
    ax[2].legend()

    ax[3].plot(x/L, Uz_hor, label=label, marker="x", markevery=2, color=pp_color)
    ax[3].plot(exp.V_left[:, 0]*1e-3, exp.V_left[:, 1]*V0*1e-3, "ko", label=exp.label)
    ax[3].set_ylabel("Uz [m/s]")
    ax[3].set_xlabel("x/L [-]")

    ax[4].plot(x, T_hor, label=label, marker="x", markevery=2, color=pp_color)
    ax[4].plot(exp.T_left[:, 0]*1e-3, exp.T_left[:, 1]+273.15, "ko", label=exp.label)
    # print(exp.T_left[:, 0]*1e-3, exp.T_left[:, 1]+273.15)
    ax[4].set_ylabel("T [K]")
    ax[4].set_xlabel("x/L [-]")
    ax[4].legend()

    ax[5].plot(
        nPlus_right[1:],
        -uPlus_right[1:],
        label="$U^+$ right",
        marker="o",
        markevery=2,
        color=pp_color,
        ls="--",
    )
    ax[5].plot(
        nPlus_left[1:],
        uPlus_left[1:],
        label="$-U^+$ left",
        marker="^",
        markevery=2,
        color=pp_color,
    )
    ax[5].plot(
        nPlus_right[1:],
        phiPlus_right[1:],
        label="$\phi^+$ right",
        marker="x",
        ls="--",
        markevery=2,
        color=pp_color,
    )
    ax[5].plot(
        nPlus_left[1:],
        phiPlus_left[1:],
        label="$\phi^+$ left",
        marker="x",
        ls="--",
        markevery=2,
        color=pp_color,
    )

    j += 1

# Make a figure legend where the color is attributed to the different postProcessing folders

ax[5].set_xscale("log")
ax[5].set_xlabel("$n^+$")
ax[5].set_ylabel("[-]")
ax[5].legend(ncol=2, frameon=True)
ax[5].set_xlim(1e-1, 1e3)
lines = [
    plt.Line2D([0], [0], color=f"C{i}", linewidth=2, linestyle="-", label=f"{folder}")
    for i, folder in enumerate(folders)
]
fig.legend(

    handles=lines,
    loc = "upper center",
    ncol=2,
    frameon=True,

)
fig.tight_layout()
# plt.figure()

# k = 0
# for ppDir in folders:
#     pp_color = f"C{k}"
#     k += 1
#     t, whf = fu.load_wallHeatFlux(f"{ppDir}/wallHeatFlux/0/wallHeatFlux.dat", start=100)
#     plt.plot(t, -whf["wall_right"], label=f"$Q_c$ {ppDir}", color=pp_color)
#     # plt.plot(t, whf["wall_left"], label=f"$Q_h$ {ppDir}")


#     peaks, _ = sp.signal.find_peaks(-whf["wall_right"], distance=1000)
#     i_start, i_end = peaks[-10], peaks[-1]
#     t_start, t_end = t[i_start], t[i_end]
#     T = t_end - t_start
#     Qc = np.trapz(-whf["wall_right"][i_start:i_end], t[i_start:i_end]) / T
#     plt.axhline(Qc, color="k", ls="--", label=f"$Q_c$ {ppDir} average")
#     # print(f"Averageing right wall from {t_start} to {t_end}")

#     # Same for left wall
#     peaks, _ = sp.signal.find_peaks(whf["wall_left"], distance=1000)
#     i_start, i_end = peaks[-10], peaks[-1]
#     t_start, t_end = t[i_start], t[i_end]
#     T = t_end - t_start
#     Qh = np.trapz(whf["wall_left"][i_start:i_end], t[i_start:i_end]) / T
#     plt.axhline(Qh, color="k", ls=":", label=f"$Q_h$ {ppDir} average")
#     # print(f"Averageing left wall from {t_start} to {t_end}")

#     plt.xlabel("t [s]")
#     plt.ylabel("Q [W/m2]")
#     plt.legend()
    
#     # print(f"{ppDir} Qc = {Qc}")
#     # print(f"{ppDir} Qh = {Qh}")

#     Nu_c = wp.Nusselt(Qc, Th - Tc, L, thermo.kappa(T0))
#     Nu_h = wp.Nusselt(Qh, Th - Tc, L, thermo.kappa(T0))
#     Nu_avg = (Nu_c + Nu_h) / 2
#     # print(f"{ppDir} Nu_c = {Nu_c}")
#     # print(f"{ppDir} Nu_h = {Nu_h}")
#     # print(f"{ppDir} Nu_avg = {Nu_avg}")





plt.tight_layout()
plt.show()

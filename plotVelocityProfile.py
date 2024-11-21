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

Ampofo = Ampofo.Ampofo()

from functions import time_average

folders = [
    "postProcessing",
    # "./cases/kOmegaSST/0.5cm0.3mm_noSource/postProcessing",
    # "./cases/kOmegaSST/1cm0.3mm_noSource/postProcessing",
    # "./cases/kOmegaSST/2cm0.3mm_noSource/postProcessing",
]

thermo = ThermophysicalProperties("constant/thermophysicalProperties")

# Read the n last velocity profiles
parameters = fu.read_parameters("parameters")
Th, Tc = parameters["T_left"], parameters["T_right"]
DeltaT = Th - Tc
T0 = (Th + Tc) / 2
L = parameters["L_x"]
V0 = np.sqrt(9.81 * L * (Th - Tc) / T0)

fig, ax = plt.subplots(2, 3, figsize=(9, 4))
ax = ax.flatten()
i = 0
for postProcessing in folders:
    print(f"Plotting {postProcessing}")
    label = postProcessing.split("/")[0]
    sorted_folders = fu.get_sorted_times(f"{postProcessing}/linesample")
    time = sorted_folders[-1]
    pp_color = f"C{i}"
    i += 1

    vertical = fu.load_sampling_set(f"{postProcessing}/linesample", time, "vertical", org=False)
    horizontal = fu.load_sampling_set(
        f"{postProcessing}/linesample", time, "horizontal", org=False)
    # right = fu.load_sampling_set(f"{postProcessing}/wallsample", time, "rightwall", org=False)
    # left = fu.load_sampling_set(f"{postProcessing}/wallsample", time, "leftwall", org=False)

    # right = fu.load(f"{postProcessing}/right_wall/{time}/line.xy")
    # left = fu.load(f"{postProcessing}/left_wall/{time}/line.xy")
    right = fu.load_sampling_set(f"{postProcessing}/right_wall", time, "line", org=False)
    left = fu.load_sampling_set(f"{postProcessing}/left_wall", time, "line", org=False)
    print(left)
    # q_right = right[:,1]
    # q_left = left[:,1]

    z = vertical["z"]  # vertical[:, 0]
    z_right = right["z"]
    z_left = left["z"]
    T_vert = vertical["TMean"]  # [:, iT]
    Ux_vert = vertical["UMeanx"]  # [:, iUx]
    Uz_vert = vertical["UMeanx"]  #:, iUz]

    q_right = right["wallHeatFluxMean"]  # [:, 1]
    q_left = left["wallHeatFluxMean"]  # [:, 1

    Nu_left = wp.Nusselt(q_left, Th - Tc, L, thermo.kappa(Th))
    Nu_right = wp.Nusselt(q_right, Th - Tc, L, thermo.kappa(Tc))

    T_hor = horizontal["TMean"]  # [:, 1]
    Uz_hor = horizontal["UMeanz"]  # [:, iUz]
    k_hor = horizontal["kMean"]  # [:, 1]
    x = horizontal["x"]  # [:, 0]

    Uz_left = horizontal["UMeanz"]  # [:, iUz]
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

    ax[1].plot(T_vert, z, label=label, marker="x", markevery=2, color=pp_color)

    Nu_left = q_left * L / ((Th - Tc) * thermo.kappa(Th))
    ax[2].plot(
        # Nu_left,
        q_left,
        z_left / L,
        label="Left wall",
        color=pp_color,
    )
    ax[2].plot(
        -Nu_right,
        z_right / L,
        label="Right wall",
        ls="--",
        color=pp_color,
    )

    ax[3].plot(x / L, Uz_hor / V0, label=label, marker="x", markevery=1, color=pp_color)

    ax[4].plot(
        x / L,
        (T_hor - Tc) / DeltaT,
        label=label,
        marker="x",
        markevery=1,
        color=pp_color,
    )

    ax[5].plot(
        x / L, k_hor / V0**2, label=label, marker="x", markevery=1, color=pp_color
    )

ax[0].set_ylabel("z/L [-]")
ax[0].set_xlabel("Ux [m/s]")

ax[1].set_ylabel("z/L [-]")
ax[1].set_xlabel("T [K]")

ax[2].plot(Ampofo.Nu_hot[:, 1], Ampofo.Nu_hot[:, 0], "xk", label=Ampofo.label + " left")
ax[2].plot(
    Ampofo.Nu_cold[:, 1], Ampofo.Nu_cold[:, 0], "ok", label=Ampofo.label + " right"
)
ax[2].set_ylabel("z/L [-]")
ax[2].set_xlabel("Nu [-]")
ax[2].legend()

ax[3].plot(Ampofo.hor_mid["X"], Ampofo.hor_mid["vBar/V0"], "ko", label=Ampofo.label)
ax[3].set_ylabel("Uz [m/s]")
ax[3].set_xlabel("x/L [-]")
ax[3].legend()

ax[4].plot(
    Ampofo.hor_mid["X"], Ampofo.hor_mid["(TBar-Tc)/DeltaT"], "ko", label=Ampofo.label
)
ax[4].set_ylabel("T [K]")
ax[4].set_xlabel("x/L [-]")
ax[4].legend()

ax[5].plot(Ampofo.hor_mid["X"], Ampofo.hor_mid["k/V0^2"], "ko", label=Ampofo.label)
ax[5].set_ylabel("k [m2/s2]")
ax[5].set_xlabel("x/L [-]")
ax[5].legend()

lines = [
    plt.Line2D([0], [0], color=f"C{i}", linewidth=2, linestyle="-", label=f"{time}")
    for i, postProcessing in enumerate(folders)
]
fig.legend(
    handles=lines,
    loc="upper center",
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

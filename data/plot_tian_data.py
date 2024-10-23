import numpy as np
import matplotlib.pyplot as plt
import scienceplots


plt.style.use(["science", "nature", "grid", "no-latex"])

Tc = 10
Th = 50

X = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1])
T_bottom = np.array([Th, 33.4, 27.76, 24.2, 21.4, 19.6, 18.4, 16.8, 14.6, Tc])
T_top = np.array([Tc, 28.8, 34.3, 36.5, 39.2, 40.8, 41.6, 42.4, 45, Th])

plt.plot(X, T_bottom, "C0o", label="T_bottom")
plt.plot(X, T_top, "C1o", label="T_top")

# Make polynomial fits
bottom_fit = np.polyfit(X, T_bottom, 7)
top_fit = np.polyfit(X, T_top, 7)

x = np.linspace(0, 1, 100)
plt.plot(x, np.polyval(bottom_fit, x), "C0", label="Bottom fit")
plt.plot(x, np.polyval(top_fit, x), "C1", label="Top fit")


plt.xlabel("X")
plt.ylabel("T")
plt.legend()
plt.show()

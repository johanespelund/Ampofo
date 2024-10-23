from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import curve_fit

plt.style.use(["science", "grid", "no-latex"])


@dataclass
class Tian:
    label: str = "Tian"
    Thot: float = 50 + 273.15
    Tcold: float = 10 + 273.15
    L: float = 0.75

    Nu_cold: np.ndarray = field(init=False)
    Nu_hot: np.ndarray = field(init=False)
    phi_bottom: np.ndarray = field(init=False)
    phi_top: np.ndarray = field(init=False)

    T_top: np.ndarray = field(init=False, default=None)
    T_bottom: np.ndarray = field(init=False, default=None)

    V_left: np.ndarray = field(init=False, default=None)
    T_left: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.Nu_cold = np.genfromtxt("data/Tian/NuColdWall.csv")
        self.Nu_hot = np.genfromtxt("data/Tian/NuHotWall.csv")
        self.phi_bottom = np.genfromtxt("data/Tian/bottomWallT.csv")
        self.phi_top = np.genfromtxt("data/Tian/topWallT.csv")
        self.V_left = np.genfromtxt("data/Tian/VelocityLeftWall.csv")
        self.T_left = np.genfromtxt("data/Tian/TemperatureLeftWall.csv")

        self.T_top = self.phi_top.copy()
        self.T_bottom = self.phi_bottom.copy()
        self.T_top[:, 1] = self.Tcold + (self.Thot - self.Tcold) * self.phi_top[:, 1]
        self.T_top[:, 0] = self.phi_top[:, 0] * self.L
        self.T_bottom[:, 1] = (
            self.Tcold + (self.Thot - self.Tcold) * self.phi_bottom[:, 1]
        )
        self.T_bottom[:, 0] = self.phi_bottom[:, 0] * self.L

    def polynomial_fit(self, x_data, y_data, degree):
        return np.polyfit(x_data, y_data, degree)

    def plot_horizontal_temperatures(self):
        poly_degree = 7
        x_fit = np.linspace(0, self.L, 100)
        T_top_fit = np.polyval(
            self.polynomial_fit(self.T_top[:, 0], self.T_top[:, 1], poly_degree), x_fit
        )
        T_bottom_fit = np.polyval(
            self.polynomial_fit(self.T_bottom[:, 0], self.T_bottom[:, 1], poly_degree),
            x_fit,
        )

        # Plot data points and fitted curve for T_top
        plt.plot(
            self.T_top[:, 0],
            self.T_top[:, 1],
            "C1s",
            label="Top data",
            markerfacecolor="none",
        )
        plt.plot(x_fit, T_top_fit, "C1--", label="Top fit")

        # Plot data points and fitted curve for T_bottom
        plt.plot(
            self.T_bottom[:, 0],
            self.T_bottom[:, 1],
            "C0o",
            label="Bottom data",
            markerfacecolor="none",
        )
        plt.plot(x_fit, T_bottom_fit, "C0-.", label="Bottom fit")

        plt.xlabel("x [m]")
        plt.ylabel("T [K]")
        plt.legend(framealpha=0.5)
        plt.tight_layout()
        plt.savefig("figures/Tian/horizontal_temperatures.svg")
        plt.show()
        self.write_poly_expression()

    def write_poly_expression(self):
        poly_degree = 7
        T_top_fit = self.polynomial_fit(self.T_top[:, 0], self.T_top[:, 1], poly_degree)
        T_bottom_fit = self.polynomial_fit(
            self.T_bottom[:, 0], self.T_bottom[:, 1], poly_degree
        )
        print(T_top_fit)
        expression_top = "expression_top\n "
        expression_bottom = "expression_bottom\n "
        # for i, coef in enumerate(T_top_fit):
        # Loop from end to start
        for i in range(poly_degree + 1):
            top_coef = T_top_fit[-i - 1]
            bottom_coef = T_bottom_fit[-i - 1]
            top_sign = "+" if top_coef >= 0 else "-"
            bottom_sign = "+" if bottom_coef >= 0 else "-"
            # expression += f"{sign} {coef:.2f}" + i*"* pos().x()" + "\n"
            expression_top += (
                f"{top_sign} {abs(top_coef):.2f}" + i * "* pos().x()" + "\n"
            )
            expression_bottom += (
                f"{bottom_sign} {abs(bottom_coef):.2f}" + i * "* pos().x()" + "\n"
            )

        with open("system/parameters.setFields", "w") as f:
            f.write(expression_top)
            f.write(expression_bottom)


if __name__ == "__main__":
    tian = Tian()
    tian.plot_horizontal_temperatures()

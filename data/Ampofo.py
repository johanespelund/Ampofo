from dataclasses import dataclass, field
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev, interp1d

plt.style.use(["science", "grid", "no-latex"])


@dataclass
class Ampofo:
    label: str = "Ampofo"
    Thot: float = 50 + 273.15
    Tcold: float = 10 + 273.15
    L: float = 0.75

    Nu_cold: np.ndarray = field(init=False)
    Nu_hot: np.ndarray = field(init=False)
    phi_bottom: np.ndarray = field(init=False)
    phi_top: np.ndarray = field(init=False)

    T_top: np.ndarray = field(init=False, default=None)
    T_bottom: np.ndarray = field(init=False, default=None)

    hor_mid: pd.DataFrame = field(init=False, default=None)

    data_path = pathlib.Path("data/Ampofo")


    def __post_init__(self):
        self.npload = lambda filename: np.genfromtxt(pathlib.Path(self.data_path, filename))
        self.pdload = lambda filename: pd.read_csv(pathlib.Path(self.data_path, filename), sep="\s+")
        self.Nu_cold = self.npload("local_nusselt_distribution.csv")[:,[0,2]]
        self.Nu_hot = self.npload("local_nusselt_distribution.csv")[:,[0,1]]
        self.phi_bottom = self.npload("horizontal_wall_temp.csv")[:,[0,2]]
        self.phi_top = self.npload("horizontal_wall_temp.csv")[:,[0,1]]
        self.hor_mid = self.pdload("Y0.5.csv")

        self.T_top = self.phi_top.copy()
        self.T_bottom = self.phi_bottom.copy()
        self.T_top[:, 1] = self.Tcold + (self.Thot - self.Tcold) * self.phi_top[:, 1]
        self.T_bottom[:, 1] = self.Tcold + (self.Thot - self.Tcold) * self.phi_bottom[:, 1]
        self.T_top[:, 0] = self.phi_top[:, 0] * self.L
        self.T_bottom[:, 0] = self.phi_bottom[:, 0] * self.L

    def polynomial_fit(self, x_data, y_data, degree):
        return np.polyfit(x_data, y_data, degree)

    def plot_horizontal_poly_temperatures(self):


        for i in range(len(self.T_top[:-1, 0])):
            x0 = self.T_top[i, 0]
            x1 = self.T_top[i+1, 0]
            T0 = self.T_top[i, 1]
            T1 = self.T_top[i+1, 1]
            # Linear interpolation
            a = (T1 - T0)/(x1 - x0)
            b = T0 - a*x0
            expression = f"x<{x1} ? {a}*x + {b} : "
            print(expression)

        for i in range(len(self.T_bottom[:-1, 0])):
            x0 = self.T_bottom[i, 0]
            x1 = self.T_bottom[i+1, 0]
            T0 = self.T_bottom[i, 1]
            T1 = self.T_bottom[i+1, 1]
            # Linear interpolation
            a = (T1 - T0)/(x1 - x0)
            b = T0 - a*x0
            expression = f"x<{x1} ? {a}*x + {b} : "
            print(expression)





        poly_degree = 8
        x_fit = np.linspace(0, self.L, 1000)
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
        plt.savefig("figures/Ampofo/horizontal_temperatures.svg")
        plt.show()
        self.write_poly_expression()

    def plot_horizontal_temperatures(self):
        x_fit = np.linspace(0, self.L, 1000)
        
        # 2nd-degree polynomial interpolation for the top temperature data
        interp_top = interp1d(self.T_top[:, 0], self.T_top[:, 1])#, kind='quadratic')
        T_top_fit = interp_top(x_fit)
        
        # 2nd-degree polynomial interpolation for the bottom temperature data
        interp_bottom = interp1d(self.T_bottom[:, 0], self.T_bottom[:, 1])#, kind='quadratic')
        T_bottom_fit = interp_bottom(x_fit)

        # Plot data points and fitted quadratic interpolation curve for T_top
        plt.plot(
            self.T_top[:, 0],
            self.T_top[:, 1],
            "C1s",
            label="Top data",
            markerfacecolor="none",
        )

        # Plot data points and fitted quadratic interpolation curve for T_bottom
        plt.plot(
            self.T_bottom[:, 0],
            self.T_bottom[:, 1],
            "C0o",
            label="Bottom data",
            markerfacecolor="none",
        )
        # plt.plot(x_fit, T_bottom_fit, "C0-.", label="Bottom quadratic fit")

        poly_degree = 8
        T_top_fit = self.polynomial_fit(self.T_top[:, 0], self.T_top[:, 1], poly_degree)
        T_bottom_fit = self.polynomial_fit(
            self.T_bottom[:, 0], self.T_bottom[:, 1], poly_degree
        )
        
        plt.plot(x_fit, np.polyval(T_top_fit, x_fit), "C1-", label="Top polynomial fit")
        plt.plot(x_fit, np.polyval(T_bottom_fit, x_fit), "C0-", label="Bottom polynomial fit")

        plt.xlabel("x [m]")
        plt.ylabel("T [K]")
        plt.legend(framealpha=0.5)
        plt.tight_layout()
        plt.savefig("figures/Ampofo/horizontal_temperatures_quadratic_interp.svg")
        plt.show()

    def write_poly_expression(self):
        poly_degree = 6
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
                f"{top_sign} {abs(top_coef):.4f}" + i * "*x" + "\n"
            )
            expression_bottom += (
                f"{bottom_sign} {abs(bottom_coef):.4f}" + i * "*x" + "\n"
            )

        with open("system/parameters.setFields", "w") as f:
            f.write(expression_top)
            f.write(expression_bottom)

def logistic(x, L=1, x_0=0, k=1):
    return L / (1 + np.exp(-k * (x - x_0)))

if __name__ == "__main__":
    Ampofo = Ampofo()
    Ampofo.plot_horizontal_poly_temperatures()

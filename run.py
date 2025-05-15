import datetime
import glob
import shutil
from subprocess import run

import click
import toml
import numpy as np
from FoamUtils import ThermophysicalProperties, read_parameters

from boundary_layer import (foam_grading, foam_grading_string,
                            replace_blocks_content, write_blocks_to_parameters)

g = 9.81


def run_command(command, dry_run=False, shell=False):
    if dry_run:
        print(
            f"Dry-run: Would execute command: {' '.join(command) if isinstance(command, list) else command}"
        )
    else:
        run(command, shell=shell)


def main(input_file=None, override=None):
    """
    Set up OpenFOAM case for simulating experiment by Ampofo & Karayiannis (2003)
    doi: 10.1016/S0017-9310(03)00147-9
    """
    parameters = {  # Default parameters
        "x_wall": 0.5e-3,
        "x_bulk": 15e-3,
        "r": 1.05,
        "highRe": False,
        "bSource": False,
        "RASModel": "laminar",
        "p_outlet": 101325,
        "n_processors": 1,
        "decompMethod": "simple",
        "nDecomp": "(2 1 2)",
        "THFM": "GGDH",
    }
    dry_run = False

    if input_file:
        parameters.update(toml.load(input_file))
    if override:
        parameters.update(override)

    LH2 = parameters["fluid"] == "H2"

    if LH2:
        parameters["L_x"] = 0.09057
        parameters["L_y"] = 0.09057
        parameters["L_z"] = 0.09057
        parameters["T_right"] = 20
        parameters["T_left"] = 30
        parameters["M"] = 2.01588
        parameters["horizontalHeatType"] = "zeroGradient"
    else:
        parameters["L_x"] = 0.75
        parameters["L_y"] = 0.75
        parameters["L_z"] = 0.75
        parameters["T_right"] = 283.15
        parameters["T_left"] = 323.15
        parameters["M"] = 28.96
        parameters["horizontalHeatType"] = "fixedValue"
    parameters["T_avg"] = (parameters["T_right"] + parameters["T_left"]) / 2

    if parameters["Ra"]:
        L = calc_L_from_Ra(parameters)
        parameters["L_x"] = L
        parameters["L_y"] = L
        parameters["L_z"] = L
        parameters["x_wall"] *= L
        parameters["x_bulk"] *= L

    # Make sure sample line doesnt go along cell edge in case of even number of cells!
    parameters["x_mid"] = np.round(L/2, 6)

    shutil.rmtree("0", ignore_errors=True)
    shutil.rmtree("postProcessing", ignore_errors=True)
    run_command(["foamListTimes", "-rm"], dry_run)
    for processor in glob.glob("processor*"):
        shutil.rmtree(processor, ignore_errors=True)

    if parameters["LTS"]:
        parameters.update(
            {
                "endTime": 50000,
                "writeInterval": 1000,
                "deltaT": 1,
                "adjustTimeStep": "no",
                "ddtScheme": "localEuler",
                "restartPeriod": 10000,
            }
        )
    else:
        parameters.update(
            {
                "endTime": 150 if LH2 else 600,
                "writeInterval": 30 if LH2 else 50,
                "deltaT": 1e-6 if LH2 else 1e-4,
                "adjustTimeStep": "yes",
                "ddtScheme": "Euler",
                # "ddtScheme": "backward",
                # "ddtScheme": "CrankNicolson 0.7",
                "restartPeriod": 30 if LH2 else 200,
            }
        )

    run_command(["cp", "system/controlDict.setup", "system/controlDict"], dry_run)

    if parameters["bSource"] and parameters["RASModel"] not in [
        "v2fBuoyant",
        "buoyantKEpsilon",
    ]:
        run_command(["cp", "constant/fvModels.bSource", "constant/fvModels"], dry_run)
        print("Buoyancy source term is included")
    else:
        run_command(["rm", "-f", "constant/fvModels"], dry_run)
        print("Buoyancy source term is not included")

    if parameters["RASModel"] == "laminar":
        parameters["simulationType"] = "laminar"
        parameters["turbulence"] = "false"
    else:
        parameters["simulationType"] = "RAS"
        parameters["turbulence"] = "true"

    parameters["Cg"] = 1 / parameters["Prt"]

    grading_params = foam_grading(
        parameters["L_x"], parameters["x_wall"], parameters["x_bulk"], parameters["r"]
    )
    n, grading = foam_grading_string(grading_params)

    v_string = f"hex (0 1 2 3 4 5 6 7)"
    n_string = f"({n} 1 {n}) // x_wall: {parameters['x_wall']} x_bulk: {parameters['x_bulk']} r: {parameters['r']}"
    grading_string = (
        f"(\n\t\t\t\t\t{grading}\n\t\t\t\t\t1\n\t\t\t\t\t{grading}\n\t\t\t\t)"
    )
    write_blocks_to_parameters(parameters, v_string, n_string, grading_string, dry_run)

    parameters["nCells"] = int(n) ** 2
    parameters["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parameters[
        "variables"
    ] = f"""
        (
            "x=pos().x()"
            "Tleft={parameters['T_left']}"
            "Tright={parameters['T_right']}"
            "L={parameters['L_x']}"
        )
    """

    write_parameters(parameters, dry_run)

    run_command(["blockMesh >> log.blockMesh"], dry_run, shell=True)
    run_command(["checkMesh >> log.checkMesh"], dry_run, shell=True)

    print(f"Created mesh of size ({n}x{n}) {parameters['nCells']} cells")

    run_command(["foamListTimes", "-rm"], dry_run)
    run_command(["rm", "-rf", "0"], dry_run)
    run_command(["cp", "-r", "0-orig", "0"], dry_run)
    run_command(["rm", "-rf", "postProcessing"], dry_run)
    if not LH2:
        print("Setting up initial conditions")
        run_command(
            [
                f"setExprBoundaryFields -dict system/setExprBoundaryFieldsDict.{parameters['horizontalBC']} >> log.setExprBoundaryFields"
            ],
            dry_run,
            shell=True,
        )

        # if parameters["horizontalBC"] == "experimental":
        #     run_command(["setExprBoundaryFields > log.setExprBoundaryFields"], dry_run, shell=True)
        # elif parameters["horizontalBC"] == "linear":
        #     run_command(["setExprBoundaryFields > log.setExprBoundaryFields"], dry_run, shell=True)
        # else:
        #     raise ValueError("Invalid horizontal boundary condition")

    if parameters["highRe"]:
        run_command("cp highReBC/* 0/", dry_run, shell=True)
        print("Using high-Re turbulent boundary conditions")
    else:
        run_command("cp lowReBC/* 0/", dry_run, shell=True)
        print("Using low-Re turbulent boundary conditions")

    # if parameters["n_processors"] > 1:
    #     run_command(["decomposePar >> log.decomposePar"], dry_run, shell=True)

    run_command(["cp", "system/controlDict.run", "system/controlDict"], dry_run)

    print("Ready to run the simulation")


def write_parameters(parameters, dry_run=False):
    with open("parameters" + ".dry-run" * dry_run, "w") as f:
        for key, value in parameters.items():
            if key not in ["date"]:
                f.write(f"{key} {value};\n")


def calc_L_from_Ra(parameters):
    thermo = ThermophysicalProperties("constant/thermophysicalProperties")
    p = parameters

    beta = thermo.beta(p["p_outlet"], p["T_avg"])
    Pr = thermo.Pr(p["T_avg"])
    nu = thermo.mu(p["T_avg"]) / thermo.rho(p["p_outlet"], p["T_avg"])
    DeltaT = abs(p["T_left"] - p["T_right"])
    g = 9.81

    # Ra = Pr*Gr = Pr*g*beta*DeltaT*L**3/(nu**2)
    # L = (Ra*nu**2/(Pr*g*beta*DeltaT))**(1/3)
    L = ((p["Ra"] * nu**2) / (Pr * g * beta * DeltaT)) ** (1 / 3)
    print(f"Calculated L: {L} m")
    Ra = Pr * g * beta * DeltaT * L**3 / (nu**2)

    if abs(Ra - p["Ra"]) / Ra > 1e-6:
        raise ValueError(f"Calculated Ra {Ra} does not match input Ra {p['Ra']}")
    return L


@click.command()
@click.option("--input-file", "-i", help="Input file")
@click.option(
    "--override",
    "-o",
    nargs=2,
    multiple=True,
    help="Override specific parameters from the input file in 'key value' pairs.",
)
def main_click(input_file, override):
    main(input_file, dict(override))


if __name__ == "__main__":
    main_click()

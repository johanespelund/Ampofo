import datetime
from subprocess import run
import shutil, glob
import toml

import click

from boundary_layer import (foam_grading, foam_grading_string,
                            replace_blocks_content,
                            write_blocks_to_parameters)
from FoamUtils import ThermophysicalProperties 
from FoamUtils import read_parameters

g = 9.81

def run_command(command, dry_run=False, shell=False):
    if dry_run:
        print(f"Dry-run: Would execute command: {' '.join(command) if isinstance(command, list) else command}")
    else:
        run(command, shell=shell)




def main(input_file=None, override=None):

    """
    Set up OpenFOAM case for simulating experiment by Ampofo & Karayiannis (2003)
    doi: 10.1016/S0017-9310(03)00147-9
    """
    parameters = {   # Default parameters
        "L_x": 0.75,
        "L_y": 0.75,
        "L_z": 0.75,
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
        "THFM": "GGDH"
    }
    dry_run = False

    if input_file:
        parameters.update(toml.load(input_file))
    if override:
        parameters.update(override)


    shutil.rmtree("0", ignore_errors=True)
    shutil.rmtree("postProcessing", ignore_errors=True)
    run_command(["foamListTimes", "-rm"], dry_run)
    for processor in glob.glob("processor*"):
        shutil.rmtree(processor, ignore_errors=True)

    parameters["T_right"] = 283.15
    parameters["T_left"] = 323.15
    parameters["T_avg"] = (parameters["T_right"] + parameters["T_left"])/2
    parameters["M"] = 28.96

    if parameters["LTS"]:
        parameters.update({
            "endTime": 50000,
            "writeInterval": 1000,
            "timeStep": 1,
            "adjustTimeStep": "no",
            "ddtScheme": "localEuler",
            "restartPeriod": 10000,
        })
    else:
        parameters.update({
            "endTime": 900,
            "writeInterval": 100,
            "timeStep": 1e-4,
            "adjustTimeStep": "yes",
            "ddtScheme": "CrankNicolson 0.75",
            "restartPeriod": 300,
        })

    run_command(["cp", "system/controlDict.setup", "system/controlDict"], dry_run)

    T0 = parameters["T_avg"]
    thermo = ThermophysicalProperties("constant/thermophysicalProperties")
    cp = thermo.Cp(T0)
    mu = thermo.mu(T0)
    kappa = thermo.kappa(T0)
    beta = 1/T0 #thermo.beta(parameters["p_outlet"], T0)
    Pr = thermo.Pr(T0)
    rho = thermo.rho(parameters["p_outlet"], T0)
    L = parameters["L_x"]
    deltaT = parameters["T_left"] - parameters["T_right"]

    print(f"{Pr=}, {rho=}, {beta=}, {mu=}, {kappa=}, {cp=}, {L=}, {deltaT=}")

    Ra = Pr*g*beta*rho**2*deltaT*L**3/(mu**2)
    print(f"Rayleigh number: {Ra: .4e}")

    def find_Th_H2(x):
        print(f"\nRunning with Th = {x[0]}")
        Th = float(x[0])
        Tc = 20
        target_Ra = 1.58e9
        T0 = (Th + Tc) / 2
        thermo = ThermophysicalProperties("constant/thermophysicalProperties")
        cp = thermo.Cp(T0)
        mu = thermo.mu(T0)
        kappa = thermo.kappa(T0)
        beta = thermo.beta(parameters["p_outlet"], T0)
        Pr = thermo.Pr(T0)
        rho = thermo.rho(parameters["p_outlet"], T0)
        rho_c = thermo.rho(parameters["p_outlet"], Tc)
        rho_h = thermo.rho(parameters["p_outlet"], Th)
        print(f"rho(Tc): {rho_c}, rho(Th): {rho_h}")
        deltaRho = thermo.rho(parameters["p_outlet"], Tc) - thermo.rho(parameters["p_outlet"], Th)
        alpha = kappa/(rho*cp)
        L = parameters["L_x"]
        deltaT = Th - Tc
        print(f"{deltaRho=}, {alpha=}, {L=}, {deltaT=}")
        Ra = deltaRho*(L**3)*g/(mu*alpha)
        print(f"{rho=}, {beta=}, {T0=}, {Pr=}, {Ra= :.4e} {Th=}, {Tc=}, {deltaT=}, {L=}, {mu=}, {kappa=}, {cp=} ")
        print(f"{Th=}, {Ra= :.4e}")
        return Ra - target_Ra

    # from scipy.optimize import fsolve
    # Th = fsolve(find_Th_H2, [50])[0]
    # print(f"Th: {Th}")
    # exit()


    if parameters["bSource"] and parameters["RASModel"] not in ["v2fBuoyant", "buoyantKEpsilon"]:
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

    grading_params = foam_grading(
        parameters["L_x"], parameters["x_wall"], parameters["x_bulk"], parameters["r"]
    )
    n, grading = foam_grading_string(grading_params)

    v_string = f"hex (0 1 2 3 4 5 6 7)"
    n_string = f"({n} 1 {n}) // x_wall: {parameters['x_wall']} x_bulk: {parameters['x_bulk']} r: {parameters['r']}"
    grading_string = f"(\n\t\t\t\t\t{grading}\n\t\t\t\t\t1\n\t\t\t\t\t{grading}\n\t\t\t\t)"
    write_blocks_to_parameters(parameters, v_string, n_string, grading_string, dry_run)

    parameters["nCells"] = int(n)**2
    parameters["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_parameters(parameters, dry_run)

    run_command(["blockMesh >> log.blockMesh"], dry_run, shell=True)

    print(f"Created mesh of size ({n}x{n}) {parameters['nCells']} cells")

    run_command(["foamListTimes", "-rm"], dry_run)
    run_command(["rm", "-rf", "0"], dry_run)
    run_command(["cp", "-r", "0-orig", "0"], dry_run)
    run_command(["rm", "-rf", "postProcessing"], dry_run)
    run_command(["setExprBoundaryFields > log.setExprBoundaryFields"], dry_run, shell=True)

    if parameters["highRe"]:
        run_command("cp highReBC/* 0/", dry_run, shell=True)
        print("Using high-Re turbulent boundary conditions")
    else:
        run_command("cp lowReBC/* 0/", dry_run, shell=True)
        print("Using low-Re turbulent boundary conditions")

    if parameters["n_processors"] > 1:
        run_command(["decomposePar >> log.decomposePar"], dry_run, shell=True)

    run_command(["cp", "system/controlDict.run", "system/controlDict"], dry_run)

    print("Ready to run the simulation")


def write_parameters(parameters, dry_run=False):
    with open("parameters"+".dry-run"*dry_run, "w") as f:
        for key, value in parameters.items():
            if key not in ["date"]:
                f.write(f"{key} {value};\n")

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


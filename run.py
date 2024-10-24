import datetime
from subprocess import run
import shutil, glob

import click

from boundary_layer import (foam_grading, foam_grading_string,
                            replace_blocks_content,
                            write_blocks_to_parameters)
from FoamUtils import ThermophysicalProperties as tp
from FoamUtils import read_parameters

g = 9.81

def run_command(command, dry_run=False, shell=False):
    if dry_run:
        print(f"Dry-run: Would execute command: {' '.join(command) if isinstance(command, list) else command}")
    else:
        run(command, shell=shell)


@click.command()
@click.option("--turbulence", "-t", is_flag=True, help="Include turbulence model")
@click.option("--map-case", "-m", default="", help="Map fields from another case to initialize")
@click.option("--model", "-m", default="kOmegaSST", help="Turbulence model")
@click.option("--buoyancy-source", "-b", is_flag=True, help="Include buoyancy source term")
@click.option("--x-wall", default=0.3e-3, help="Wall cell size")
@click.option("--x-bulk", default=0.01, help="Bulk cell size")
@click.option("--n-processors", default=1, help="Number of processors")
@click.option("--decomp-method", "-d", default="scotch", type=click.Choice(["simple", "scotch", "hierarchical"]), help="Decomposition method")
@click.option("--n-decomp", "-n", default="(1 1 1)", help="Coeffs for simple and hierarchical decomposition")
@click.option("--config-file", "-c", default="", help="Configuration file")
@click.option("--dry-run", is_flag=True, help="Print actions without modifying the system")
@click.option("--yes-clean", "-y", is_flag=True, help="Skip confirmation and clean the system")
def main(turbulence, map_case, model, buoyancy_source, x_wall, x_bulk, n_processors, decomp_method, n_decomp, config_file, dry_run, yes_clean):
    """
    Set up OpenFOAM case for simulating experiment by Ampofo & Karayiannis (2003)
    doi: 10.1016/S0017-9310(03)00147-9
    """
    parameters = {
        "L_x": 0.75,
        "L_y": 0.75,
        "L_z": 0.75,
        "map_case": map_case,
        "x_wall": x_wall,
        "x_bulk": x_bulk,
        "r": 1.1,
        "bSource": buoyancy_source,
        "turbulence": turbulence,
        "RASModel":model,
        "p_outlet": 101325,
        "n_processors": n_processors,
        "decompMethod": decomp_method,
        "nDecomp": n_decomp,
    }

    a = False
    if dry_run:
        print("Dry-run mode enabled. No system changes will be made.")
    elif not yes_clean:
        a = input("This will overwrite the current system. Are you sure you want to continue? (y/n): ")
        a = a.lower() == "y"
    if a or yes_clean:
        shutil.rmtree("0", ignore_errors=True)
        shutil.rmtree("postProcessing", ignore_errors=True)
        run_command(["foamListTimes", "-rm"], dry_run)
        for processor in glob.glob("processor*"):
            shutil.rmtree(processor, ignore_errors=True)
    else:
        print("Exiting without making any changes")
        return
        

    parameters["T_right"] = 283.15
    parameters["T_left"] = 323.15
    parameters["T_avg"] = (parameters["T_right"] + parameters["T_left"])/2
    parameters["M"] = 28.96

    if config_file:
        print("Reading parameters from", config_file)
        parameters.update(read_parameters(config_file))

    T0 = parameters["T_avg"]
    thermo = tp.ThermophysicalProperties("constant/thermophysicalProperties")
    cp = thermo.Cp(T0)
    mu = thermo.mu(T0)
    kappa = thermo.kappa(T0)
    beta = thermo.beta(parameters["p_outlet"], T0)
    Pr = thermo.Pr(T0)
    rho = thermo.rho(parameters["p_outlet"], T0)
    L = parameters["L_x"]
    deltaT = parameters["T_left"] - parameters["T_right"]

    Ra = Pr*g*beta*rho**2*deltaT*L**3/(mu**2)
    print(f"Rayleigh number: {Ra: .4e}")

    if parameters["bSource"].lower() == "true":
        run_command(["cp", "constant/fvOptions.bSource", "constant/fvOptions"], dry_run)
        print("Buoyancy source term is included")
    else:
        run_command(["rm", "-f", "constant/fvOptions"], dry_run)
        print("Buoyancy source term is not included")

    if not parameters["turbulence"] or parameters["turbulence"] == "laminar":
        parameters["simulationType"] = "laminar"
        parameters["turbulence"] = "false"
        parameters["RASModel"] = "laminar"
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

    if parameters["map_case"]:
        raise NotImplementedError("Mapping fields from another case is not yet implemented")

    run_command(["decomposePar >> log.decomposePar"], dry_run, shell=True)

    print("Ready to run the simulation")


def write_parameters(parameters, dry_run=False):
    with open("parameters"+".dry-run"*dry_run, "w") as f:
        for key, value in parameters.items():
            if key not in ["date"]:
                f.write(f"{key} {value};\n")


if __name__ == "__main__":
    main()


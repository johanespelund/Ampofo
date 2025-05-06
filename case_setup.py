from run import main as run_main
import datetime
import toml
import pathlib
from pathlib import Path
import shutil
from subprocess import run

DATE = datetime.datetime.now().strftime("%Y-%m-%d")
DATETIME = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
CASES_DIR =  Path("cases") / "DNS_Ra1e10" / "no_buoyancy_source" 
SHARED_SLURM_TEMPLATE = Path("run-shared.slurm")

def main():

    input_file = "parameters.toml"
    input_parameters = {
        # "RASModel": ["kOmegaSST"] * 3,
        "RASModel": ["kOmegaSST", "LaunderSharmaKE", "v2f", "laminar"],
        # "x_bulk": [15e-3, 10e-3, 6.67e-3],
        # "maxCo": [0.1, 0.2, 0.4],
        # "nDecomp": ["(2 1 2)", "(3 1 4)", "(4 1 4)"],
        "nDecomp": ["(2 1 4)"] * 4,
        "LTS": [False] * 4,
        "bSource": [False] * 4,
        # "THFM": ["GGDH"] * 4,
    }
    N_sim = 4 # Must equatl the number of elements in the list(s) above

    var_params = [k for k, v in input_parameters.items() if isinstance(v, list)]
    input_dict = toml.load(input_file)
    cases = []
    n_processors = ["8"] * 4 #, "4", "8", "12"]
    # n_processors = ["4", "12", "16"]


    for i in range(N_sim):
        for var in var_params:
            input_dict[var] = input_parameters[var][i]
        input_dict["n_processors"] = n_processors[i]

        case_name = create_case_name(input_dict)
        cases.append(str(Path(CASES_DIR,case_name).resolve()))
        run_main(input_file=None, override=input_dict)
        copy_case(case_name)

    run_slurm(cases, n_processors)

def create_case_name(parameters_dict):
    p = parameters_dict
    return f"{DATE}_Ampofo_{p['RASModel']}_W{p['x_wall']}_B{p['x_bulk']}_Bts{p['bSource']}_LTS-{p['LTS']}_maxCo{p['maxCo']}"

def run_slurm(cases, n_processors):
    with open(SHARED_SLURM_TEMPLATE) as f:
        lines = f.readlines()
    lines = [l.replace("<JOB_NAME>", f"Ampofo-{DATETIME}") for l in lines]
    lines = [l.replace("<OUTPUT_DIR>", str(CASES_DIR / f"shared-{DATETIME}")) for l in lines]
    lines = [l.replace("<CASES>", " ".join(cases)) for l in lines]
    lines = [l.replace("<N_PROCESSORS>", " ".join(n_processors)) for l in lines]

    with open(CASES_DIR / f"run-shared-{DATETIME}.slurm", "w") as f:
        f.writelines(lines)

    run(["sbatch", str(CASES_DIR / f"run-shared-{DATETIME}.slurm")])


def copy_case(case_name):
    target_path = CASES_DIR / case_name
    target_path.mkdir(parents=True, exist_ok=True)

    # Copy 0, constant, system, parameters, pareters.toml, case_setup.py
    for d in ["0", "constant", "system"]:
        shutil.copytree(d, target_path / d)
    for f in ["parameters", "parameters.toml", "case_setup.py"]: #, f"run-shared-{DATETIME}.slurm"]:
        shutil.copy(f, target_path / f)


if __name__ == "__main__":
    main()

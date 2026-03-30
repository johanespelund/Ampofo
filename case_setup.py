from run import main as run_main
import datetime
import toml
import pathlib
import shutil
from pathlib import Path
from subprocess import run

DATE = datetime.datetime.now().strftime("%Y-%m-%d")
DATETIME = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
# CASES_DIR =  Path("cases") /   "DNS_Ra1e10" / "test-v2f-source-k-eps-posTanHEpsSource/Boussinesq-DeltaT10K-CpEqCv" #/ "allEqnSource-SebilleuEps-THFM" 
# CASES_DIR =  Path("cases") / "Boussinesq-deltaT60K" / "tanhEpsSource-SuSpk-v2" / "Ra1e11"
CASES_DIR =  Path("cases") / "Boussinesq-deltaT40K" / "SST-highRe" / "Ra1e11"
SHARED_SLURM_TEMPLATE = Path("run-shared.slurm")

def main():

    input_file = "parameters.toml"
    input_parameters = {
        "RASModel": ["kOmegaSST"]*2,
        "x_wall": [2.5e-3]*2,
        "x_bulk": [4.5e-3]*2,
        "Ra": [1e11]*2,
        # "x_bulk": [13.5e-3, 9e-3]*2,
        "bSource": [True]*2,
        "THFM": ["SGDH"]*2, #, "GGDH"],
        "highRe": [True]*2, #, "GGDH"],
        "Prt": [0.85, 0.85],
        "nDecomp": ["(6 1 6)"],
        # "nDecomp": ["(4 1 4)"]*2,
        # "nDecomp": ["(3 1 3)", "(4 1 5)"],
        "maxCo": [0.5]*2,
        "consistent": [True, True],
        "nOuterCorrectors": [5, 5],

        ## For grid convergence study
        # "RASModel": ["v2f"]*3,
        # "x_bulk": [12e-3, 6e-3, 3e-3],
        # "nDecomp": ["(2 1 2)", "(3 1 4)", "(4 1 4)"],
        # "bSource": [False]*3,
        # "THFM": ["SGDH"]*3 ,
    }
    N_sim = len(input_parameters["nDecomp"])  # Must equal the number of elements in the list(s) above

    var_params = [k for k, v in input_parameters.items() if isinstance(v, list)]
    input_dict = toml.load(input_file)
    cases = []
    n_processors = [get_n_proc(nd) for nd in input_parameters["nDecomp"]]

    for i in range(N_sim):
        for var in var_params:
            input_dict[var] = input_parameters[var][i]
        input_dict["n_processors"] = n_processors[i]

        case_name = create_case_name(input_dict)
        cases.append(str(Path(CASES_DIR,case_name).resolve()))
        run_main(input_file=None, override=input_dict)
        copy_case(case_name)

    run_slurm(cases, n_processors)


def get_n_proc(decomp_string):
    dirs = [float(n) for n in decomp_string.replace("(","").replace(")","").split()]
    return str(int(dirs[0] * dirs[1] * dirs[2]))


def create_case_name(parameters_dict):
    p = parameters_dict
    return f"{DATE}_Ampofo_{p['RASModel']}_W{p['x_wall']}_B{p['x_bulk']}_Bts{p['bSource']}-{p['THFM']}_Prt-{p['Prt']}_maxCo{p['maxCo']}"

def run_slurm(cases, n_processors):
    with open(SHARED_SLURM_TEMPLATE) as f:
        lines = f.readlines()
    lines = [l.replace("<JOB_NAME>", f"Ampofo-{DATETIME}") for l in lines]
    lines = [l.replace("<OUTPUT_DIR>", cases[-1]) for l in lines]
    lines = [l.replace("<CASES>", " ".join(cases)) for l in lines]
    lines = [l.replace("<N_PROCESSORS>", " ".join(n_processors)) for l in lines]

    sbatch_file = Path(cases[-1]) / f"run-{DATETIME}.slurm"
    with open(sbatch_file, "w") as f:
        f.writelines(lines)

    # run(["sbatch", "-n", str(n_processors[0]), "--exclude", "c1-6", "--exclude", "c1-2",   str(CASES_DIR / f"run-shared-{DATETIME}.slurm")])
    run(["sbatch", "-n", str(n_processors[0]), sbatch_file])


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

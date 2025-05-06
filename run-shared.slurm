#!/bin/bash
#SBATCH --account=nn11070k
#SBATCH --job-name=<JOB_NAME>
#SBATCH --output=<OUTPUT_DIR>/<JOB_NAME>.out
#SBATCH --error=<OUTPUT_DIR>/<JOB_NAME>.err
#SBATCH --ntasks-per-node=32
#SBATCH --time=2-00:00:00

echo "Purging modules"
module purge
echo "Loading OpenFOAM 12"
module load OpenFOAM/12-foss-2023a
echo "Sourcing foam bashrc: $FOAM_BASH"
source $FOAM_BASH

echo "Will use the foamRun solver: " $(which foamRun)
ls -l $(which foamRun)
ldd $(which foamRun)

echo "Starting the loop"


# Define cases and processor counts
cases=(<CASES>)
n_processors=(<N_PROCESSORS>)

for i in "${!cases[@]}"; do
    case_dir="${cases[i]}"
    n_proc="${n_processors[i]}"

    cd "$case_dir"

    decomposePar > log.decomposePar

    echo "Starting simulation in $case_dir with $n_proc processors"
    srun --exclusive -n "$n_proc" foamRun -parallel > /dev/null 2>log.errors &

    cd -
done

wait

echo "All cases completed."


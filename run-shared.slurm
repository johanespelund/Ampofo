#!/bin/bash
#SBATCH --account=nn11070k
#SBATCH --job-name=<JOB_NAME>
#SBATCH --output=<OUTPUT_DIR>/<JOB_NAME>.out
#SBATCH --error=<OUTPUT_DIR>/<JOB_NAME>.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=4G
#SBATCH -N 1

echo "Purging modules"
module purge
echo "Loading OpenFOAM 12"
module load EESSI/2023.06
module load OpenFOAM/12-foss-2023a
echo "Sourcing foam bashrc: $FOAM_BASH"
source $FOAM_BASH

echo "Will use the foamRun solver: " $(which foamRun)

echo "Starting the loop"


# Define cases and processor counts
cases=(<CASES>)
n_processors=(<N_PROCESSORS>)

for i in "${!cases[@]}"; do
    case_dir="${cases[i]}"
    n_proc="${n_processors[i]}"

    cd "$case_dir"


    # mapFields /cluster/projects/nn11070k/johan-test/Ampofo/cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-14_Ampofo_v2fBuoyant_W0.0005_B0.004_BtsTrue-SGDH_Prt-0.85_maxCo0.5/ -consistent -parallelSource -sourceTime 800 > log.mapFields 2>log.errors

    # mapFields /cluster/projects/nn11070k/johan-test/Ampofo/cases/Boussinesq-deltaT40K/SST-highRe/Ra1e11/2026-02-20_Ampofo_kOmegaSST_W0.0025_B0.0045_BtsTrue-SGDH_Prt-0.85_maxCo0.5/ -parallelSource -sourceTime 800 > log.mapFields 2>log.errors
    #
    # if [ $? -ne 0 ]; then
    #     echo "mapFields failed!"
    #     exit
    # fi

    decomposePar > log.decomposePar 2>>log.errors

    echo "Starting simulation in $case_dir with $n_proc processors"
    srun  -n "$n_proc" foamRun -parallel > /dev/null 2>>log.errors &
    # srun -n "$n_proc" foamRun -parallel > log.foamRun 2>&1 &

    cd -
done

wait

echo "All cases completed."


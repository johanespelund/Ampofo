#!/bin/bash
#SBATCH --account=nn11070k
#SBATCH --time=0-4:0:0
#SBATCH --nodes=1
#SBATCH --output=run.out

# Safety settings
set -o errexit
# set -o nounset

module --quiet purge
module load OpenFOAM/v2312-foss-2023a
source $FOAM_BASH

srun buoyantPimpleFoam -parallel > log.buoyantPimpleFoam


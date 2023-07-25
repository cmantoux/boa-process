#!/bin/bash
#SBATCH --partition=[TO BE DETERMINED]
#SBATCH --time=2:00:00
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --chdir=[TO BE DETERMINED]
#SBATCH --output="logs/%A_%a.txt"
#SBATCH --array=0-29

python E7.py -s $1 $2 $3 $4 $5 $6 ${SLURM_ARRAY_TASK_ID}

#!/bin/bash
#SBATCH --job-name="ml"
#SBATCH --partition=rtx2080ti
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:10
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt
###SBATCH --test-only

sbatch_pre.sh
module load gcc/8.3.0 cuda/11.0 python/3.8.7-gnu-gpu
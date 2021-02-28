#!/bin/bash
#SBATCH --job-name="ml"
#SBATCH --partition=rtx2080ti
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:10
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt
###SBATCH --test-only

sbatch_pre.sh
module load gcc/8.3.0 cuda/11.0 python/3.8.7-gnu-gpu
python3 classify.py Phase
python3 classify.py Bands
python3 classify.py Error
python3 classify.py MAD
python3 classify.py BS
python3 classify.py BC
python3 classify.py Orient
python3 classify.py Bands_Phase
python3 classify.py Error_Phase
python3 classify.py Error_Bands
python3 classify.py MAD_Phase
python3 classify.py MAD_Bands
python3 classify.py MAD_Error
python3 classify.py BS_Phase
python3 classify.py BS_Bands
python3 classify.py BS_Error
python3 classify.py BS_MAD
python3 classify.py BC_Phase
python3 classify.py BC_Bands
python3 classify.py BC_Error
python3 classify.py BC_MAD
python3 classify.py BC_BS
python3 classify.py Orient_Phase
python3 classify.py Orient_Bands
python3 classify.py Orient_Error
python3 classify.py Orient_MAD
python3 classify.py Orient_BS
python3 classify.py Orient_BC
python3 classify.py Error_Bands_Phase
python3 classify.py MAD_Bands_Phase
python3 classify.py MAD_Error_Phase
python3 classify.py MAD_Error_Bands
python3 classify.py BS_Bands_Phase
python3 classify.py BS_Error_Phase
python3 classify.py BS_Error_Bands
python3 classify.py BS_MAD_Phase
python3 classify.py BS_MAD_Bands
python3 classify.py BS_MAD_Error
python3 classify.py BC_Bands_Phase
python3 classify.py BC_Error_Phase
python3 classify.py BC_Error_Bands
python3 classify.py BC_MAD_Phase
python3 classify.py BC_MAD_Bands
python3 classify.py BC_MAD_Error
python3 classify.py BC_BS_Phase
python3 classify.py BC_BS_Bands
python3 classify.py BC_BS_Error
python3 classify.py BC_BS_MAD
python3 classify.py Orient_Bands_Phase
python3 classify.py Orient_Error_Phase
python3 classify.py Orient_Error_Bands
python3 classify.py Orient_MAD_Phase
python3 classify.py Orient_MAD_Bands
python3 classify.py Orient_MAD_Error
python3 classify.py Orient_BS_Phase
python3 classify.py Orient_BS_Bands
python3 classify.py Orient_BS_Error
python3 classify.py Orient_BS_MAD
python3 classify.py Orient_BC_Phase
python3 classify.py Orient_BC_Bands
python3 classify.py Orient_BC_Error
python3 classify.py Orient_BC_MAD
python3 classify.py Orient_BC_BS
python3 classify.py MAD_Error_Bands_Phase
python3 classify.py BS_Error_Bands_Phase
python3 classify.py BS_MAD_Bands_Phase
python3 classify.py BS_MAD_Error_Phase
python3 classify.py BS_MAD_Error_Bands
python3 classify.py BC_Error_Bands_Phase
python3 classify.py BC_MAD_Bands_Phase
python3 classify.py BC_MAD_Error_Phase
python3 classify.py BC_MAD_Error_Bands
python3 classify.py BC_BS_Bands_Phase
python3 classify.py BC_BS_Error_Phase
python3 classify.py BC_BS_Error_Bands
python3 classify.py BC_BS_MAD_Phase
python3 classify.py BC_BS_MAD_Bands
python3 classify.py BC_BS_MAD_Error
python3 classify.py Orient_Error_Bands_Phase
python3 classify.py Orient_MAD_Bands_Phase
python3 classify.py Orient_MAD_Error_Phase
python3 classify.py Orient_MAD_Error_Bands
python3 classify.py Orient_BS_Bands_Phase
python3 classify.py Orient_BS_Error_Phase
python3 classify.py Orient_BS_Error_Bands
python3 classify.py Orient_BS_MAD_Phase
python3 classify.py Orient_BS_MAD_Bands
python3 classify.py Orient_BS_MAD_Error
python3 classify.py Orient_BC_Bands_Phase
python3 classify.py Orient_BC_Error_Phase
python3 classify.py Orient_BC_Error_Bands
python3 classify.py Orient_BC_MAD_Phase
python3 classify.py Orient_BC_MAD_Bands
python3 classify.py Orient_BC_MAD_Error
python3 classify.py Orient_BC_BS_Phase
python3 classify.py Orient_BC_BS_Bands
python3 classify.py Orient_BC_BS_Error
python3 classify.py Orient_BC_BS_MAD
python3 classify.py BS_MAD_Error_Bands_Phase
python3 classify.py BC_MAD_Error_Bands_Phase
python3 classify.py BC_BS_Error_Bands_Phase
python3 classify.py BC_BS_MAD_Bands_Phase
python3 classify.py BC_BS_MAD_Error_Phase
python3 classify.py BC_BS_MAD_Error_Bands
python3 classify.py Orient_MAD_Error_Bands_Phase
python3 classify.py Orient_BS_Error_Bands_Phase
python3 classify.py Orient_BS_MAD_Bands_Phase
python3 classify.py Orient_BS_MAD_Error_Phase
python3 classify.py Orient_BS_MAD_Error_Bands
python3 classify.py Orient_BC_Error_Bands_Phase
python3 classify.py Orient_BC_MAD_Bands_Phase
python3 classify.py Orient_BC_MAD_Error_Phase
python3 classify.py Orient_BC_MAD_Error_Bands
python3 classify.py Orient_BC_BS_Bands_Phase
python3 classify.py Orient_BC_BS_Error_Phase
python3 classify.py Orient_BC_BS_Error_Bands
python3 classify.py Orient_BC_BS_MAD_Phase
python3 classify.py Orient_BC_BS_MAD_Bands
python3 classify.py Orient_BC_BS_MAD_Error
python3 classify.py BC_BS_MAD_Error_Bands_Phase
python3 classify.py Orient_BS_MAD_Error_Bands_Phase
python3 classify.py Orient_BC_MAD_Error_Bands_Phase
python3 classify.py Orient_BC_BS_Error_Bands_Phase
python3 classify.py Orient_BC_BS_MAD_Bands_Phase
python3 classify.py Orient_BC_BS_MAD_Error_Phase
python3 classify.py Orient_BC_BS_MAD_Error_Bands
python3 classify.py Orient_BC_BS_MAD_Error_Bands_Phase


sbatch_post.sh


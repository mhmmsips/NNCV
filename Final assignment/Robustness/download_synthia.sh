#!/bin/bash
#SBATCH --job-name=download_synthia
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G



#NOTE: NO NEED TO DO THIS OVER SLURM. JUST INCLUDE THE DOWNLOAD IN THE README. Delete the non-rgb files/directories. Rename rgb to synthia and place in data folder.

export PATH=$PATH:/gpfs/home2/scur2420/.local/bin

kaggle datasets download -d tzokas027/synthia --unzip -p "/home/scur2420/NNCV/Final assignment/"
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00


srun apptainer exec --nv --env-file .env container_with_accelerate2.sif /bin/bash main.sh

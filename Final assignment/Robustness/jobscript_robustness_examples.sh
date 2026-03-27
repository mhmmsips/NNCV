#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00


srun apptainer exec --nv --env-file .env container_with_accelerate.sif /bin/bash main_robustness_examples.sh

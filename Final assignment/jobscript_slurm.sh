#!/bin/bash
#SBTACH --nodes=1
#SBTACH --ntasks=1
#SBTACH --cpus-per-task=18
#SBTACH --gpus=1
#SBTACH --partition=gpu_a100
#SBTACH --time=00:30:00

set -a
source .env

srun apptainer exec --nv --env-file .env $CONTAINER /bin/bash main.sh
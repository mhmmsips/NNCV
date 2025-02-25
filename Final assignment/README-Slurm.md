# Running Jobs on the SLURM Cluster  

This repository includes scripts to help you get started with running your models on a SLURM cluster. Follow the steps below to set up your environment, configure your API keys, and submit a job to the server.  

---

## Step 1: Clone the Repository  

Begin by cloning the repository to the HPC cluster and navigating into the appropriate directory:  

```bash
git clone "https://<your-username>:<your-api-key>@github.com/<your-username>/<your-repo-name>"
cd "<your-repo-name>/Final assignment"
```
Replace `<your-username>` and `<your-repo-name>` with your GitHub username and the name of your repository.

## Step 2: Download the Prerequisites

We have stored all the necessary training and validation data, alongside with a Docker container that contains all the necessary modules, on a Huggingface page. To download these into the directory, submit the following script to SLURM and wait until the job has finished. This should leave you with a folder `data` and a `container.sif` file.

```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

> Note that we first add execution rights to the file to avoid any errors. You only have to do this once.

## Step 3: Configure Paths and API Keys

The `.env` file in this repository is used to set up environment variables required for your job. Update this file to include the correct paths and your API keys:

1. Open the `.env` file using a text editor:

   ```bash
   nano .env
   ```

   If you are using MobaXTerm, you can also just open it using the file explorer windown on the right side.

2. Update the following variables:

   - `WANDB_API_KEY`: Your Weights & Biases API key (for logging experiments).
   - `WANDB_DIR`: Path to the directory where the logs will be stored.

3. Save and exit the file.

## Step 4: Submit a Job to the Cluster

You will use the `jobscript_slurm.sh` file to submit a job to the SLURM cluster. This script specifies the resources and commands needed to execute your training. In our case, the file executes another bash cript, `main.sh`, inside the container you just downloaded.

Submit the job with the following command:

```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

Once submitted, SLURM will schedule your job, and the cluster will handle the execution.

---

## Explaination of SLURM Parameters

The `jobscript_slurm.sh` file includes several SLURM-specific directives (denoted by #SBATCH). Here’s a brief explanation of these commands:

- `#SBATCH --nodes=1`  
   Specifies the number of nodes (computers) your job will use. Here, only one node is requested.
- `#SBATCH --ntasks=1`  
   Specifies the number of tasks (processes) for the job. In this case, a single task is requested.
- `#SBATCH --cpus-per-task=18`  
   Allocates 18 CPU cores for the task. This value should match the requirements of your workload.
- `#SBATCH --gpus=1`  
   Requests one GPU for the job.
- `#SBATCH --partition=gpu_a100`  
   Specifies the partition to run the job on. gpu_a100 refers to a partition with NVIDIA A100 GPUs.
- `#SBATCH --time=00:30:00`  
   Sets a time limit of 30 minutes for the job. Adjust this value based on your expected runtime.

---

## Understanding the Scripts

`jobscript_slurm.sh`

This is the job submission script. It runs the `main.sh` script inside the specified container using `apptainer exec`.

```bash
#!/bin/bash  
#SBATCH --nodes=1  
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=18  
#SBATCH --gpus=1  
#SBATCH --partition=gpu_a100  
#SBATCH --time=00:30:00  

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh
```

Here you can make changes to the requested hardware for your run. On the [website](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting) of SURF, you can find how many credits each partition will cost you per allocation. Make sure you request enough time for the script to finish, as SLURM will force your script to stop after the specified time has passed. However, don't set the time limit too high, as longer script will take longer to schedule.

`main.sh`

This script contains the commands to execute inside the container. It:

1. Logs in to Weights & Biases (W&B) for experiment tracking.
2. Runs the training script (`train.py`), which is configured for single-gpu training.
3. Parses the desired hyperparameters to the `ArgumentParser`.

```bash
wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 4 \
    --seed 42 \
    --experiment-id "unet-training" \
```

If you need to make any changes to the input arguments of your script (e.g., change the `experiment-id` to avoid you are overwriting the results of your previous experiment), this is the place to be.

---

## Notes

- **Monitor your job**: Use the `squeue` command to check the status of your submitted jobs.
- **Check logs**: SLURM will create log files (`slurm-<job_id>.out`) where you can see the output of your job.
- **Adjust resources**: Modify the SLURM parameters in `jobscript_slurm.sh` to suit your task’s resource requirements.
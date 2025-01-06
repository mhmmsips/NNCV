# Final assignment: Cityscape Challenge  

Welcome to the **Cityscape Challenge**, the final project for this course!  

In this assignment, you'll put your knowledge of Neural Networks (NNs) for computer vision into action by tackling real-world problems using the **CityScapes dataset**. This dataset contains large-scale, high-quality images of urban environments, making it perfect for tasks like **semantic segmentation** and **object detection**.  

This challenge is designed to push your skills further, focusing on practical and often under-explored issues crucial for deploying computer vision models in real-world scenarios.  

---

## Benchmarks  

The competition comprises four benchmarks, each targeting a specific aspect of model performance:  

1. **Peak performance**  
   This benchmark serves as your baseline. It evaluates the model's segmentation accuracy on a clean, standard test set. Your goal is to achieve the highest segmentation scores here.  

2. **Robustness**  
   This benchmark tests how well your model performs under challenging conditions, such as changes in lighting, weather, or image quality. Consistency is key in this category.  

3. **Efficiency**  
   Practical applications often require compact models. This benchmark emphasizes creating smaller models that maintain acceptable performance. It’s particularly relevant for edge devices where large models are infeasible.  

4. **Out-of-distribution detection**  
   Models often encounter data that differs from the training distribution, leading to unreliable predictions. This benchmark evaluates your model's ability to detect and handle such out-of-distribution samples.  

---

## Deliverables  

Your final submission will consist of the following:  

### 1. Research paper  
Write a **4-page research paper** in IEEE double-column format, addressing the following:  

- **Introduction**: Present the problem, challenges, and potential solutions based on existing literature.  
- **Baseline implementation**: Describe your baseline approach and results using an off-the-shelf segmentation model.  
- **Improvements**: Outline the enhancements you made, supported by experimental results and justifications.  
- **Discussion**: Discuss limitations and suggest future improvements.  
- **Figures & tables**: Use clear visuals to support your findings.

> **Submission**: Submit your paper as a PDF document via **Canvas**.

The paper will be graded based on clarity, experimental design, insight, and originality.  

### 2. Code repository  
Push all relevant code to a **public GitHub repository** with a README.md file detailing:  
- Required libraries and installation instructions.  
- Steps to run your code.  
- Your Codalab username and TU/e email address for correct mapping across systems.  

### 3. Codalab submissions  
Submit your model for evaluation to the **Codalab challenge server**, which includes four benchmark test sets.  

---

## Grading and bonus points  

The final assignment accounts for **50% of your course grade**. Additionally, bonus points are available:  

- **Top 3 in any benchmark**: +0.25 to your final assignment grade.  
- **Best performance in any benchmark**: +0.5 to your final assignment grade.  

For example, achieving the best performance in 'Peak Performance' and a top 3 spot in another benchmark will earn you a 0.75 bonus.  

> **Note**: The bonus is optional. A great report with an innovative solution that doesn't rank highly can still earn a perfect score (10).  

---

## Important notes  

- Ensure a proper **train-validation split** of the CityScapes dataset.  
- Training your model may take many hours; plan accordingly.  
- Use ideas from literature but remember to **cite all sources**. Plagiarism will not be tolerated.  
- For questions or challenges, use the **Discussions** section of this repository to collaborate with peers.  

---

We wish you the best of luck in this challenge and are excited to see the innovative solutions you develop! 🚀

---

## Running jobs on the SLURM cluster  

This repository includes scripts to help you get started with running your models on a SLURM cluster. Follow the steps below to set up your environment, configure your API keys, and submit a job to the server.  

---

### Step 1: Clone the repository  

Begin by cloning the repository to your local system and navigating into the appropriate directory:  

```bash  
git clone https://github.com/TUE-VCA/NNCV  
cd NNCV/final_assignment  
```

### Step 2: Configure paths and API keys

The `.env` file in this repository is used to set up environment variables required for your job. Update this file to include the correct paths and your API keys:

1. Open the `.env` file using a text editor:
   ```bash
   nano .env
   ```
2. Update the following variables:
   - `CONTAINER`: Path to the container you'll use (e.g., an Apptainer/Singularity image).
   - `WANDB_API_KEY`: Your Weights & Biases API key (for logging experiments).
   - `WANDB_DIR`: Path to the directory where the logs will be stored.
3. Save and exit the file.

### Step 3: Submit a job to the cluster

You will use the `jobscript_slurm.sh` file to submit a job to the SLURM cluster. This script specifies the resources and commands needed to execute your training.

Submit the job with the following command:

```bash
sbatch jobscript_slurm.sh
```

Once submitted, the cluster will schedule your job, and SLURM will handle the execution.

### Explaination of SLURM parameters

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

### Understanding the scripts

`jobscript_slurm.sh`

This is the job submission script. It:

1. Sources the `.env` file to load environment variables.
2. Runs the `main.sh` script inside the specified container using `apptainer exec`.

```bash
#!/bin/bash  
#SBATCH --nodes=1  
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=18  
#SBATCH --gpus=1  
#SBATCH --partition=gpu_a100  
#SBATCH --time=00:30:00  

set -a  
source .env  

srun apptainer exec --nv --env-file .env $CONTAINER /bin/bash main.sh
```

`main.sh`

This script contains the commands to execute inside the container. It:

Logs in to Weights & Biases (W&B) for experiment tracking.
Runs the training script (`train.py`) using `torchrun`, which is configured for distributed training.

```bash
wandb login  

torchrun --nnodes=1 -nproc_per_node=1 train.py \  
    --data-dir /data/Cityscapes
```

### Notes

- **Monitor your job**: Use `squeue` to check the status of your submitted job.
- **Check logs**: SLURM will create log files (`slurm-<job_id>.out`) where you can see the output of your job.
- **Adjust resources**: Modify the SLURM parameters in `jobscript_slurm.sh` to suit your task’s resource requirements.
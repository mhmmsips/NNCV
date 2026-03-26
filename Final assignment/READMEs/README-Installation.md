# Getting Started Guide
Welcome to the `Final Assignment` repository! This guide will help you set up your environment and tools necessary for your work on our project using a remote High-Performance Computing (HPC) cluster.

## Environment Setup
### Visual Studio Code (VSCode)
We'll use VSCode as our Integrated Development Environment (IDE) for Python development. VSCode offers:
- **Lightweight and Fast**: A robust editor without being resource-heavy.
- **Extensions**: Support for Python, Git, SSH, and many other extensions.
- **Integrated Terminal**: Perform cluster operations and run scripts without leaving the editor.
- **Remote Development**: With extensions like "Remote - SSH," you can edit files directly on the cluster.

If you haven't installed it yet, you can download it [here](https://code.visualstudio.com/download).

### GitHub
We will be hosting our code on GitHub. GitHub is a cloud-based platform for version control and collaboration, allowing us to:
- **Manage Code Versions**: Keep track of all changes to the codebase, so we can revert to earlier versions if needed.
- **Collaborate**: Multiple team members can work on the same codebase simultaneously without conflicts.
- **Backup Code Securely**: Store your code on a remote server to avoid losing work due to local machine issues.
- **Integrate with Tools**: GitHub integrates seamlessly with CI/CD tools, testing frameworks, and more to streamline development.

If you don't have an account yet, you can [sign up here](https://github.com/join). Also make sure to have Git installed on your local system. ([Download here!](https://git-scm.com/downloads))

### Weights and Biases (W&B)
Weights and Biases (W&B) is a powerful tool for experiment tracking and visualization, especially useful in cluster environments where resources are shared, and reproducibility is critical. Here's why W&B is invaluable:
- **Track Training Metrics**: Automatically log your model's performance, loss curves, and other metrics in real-time.
- **Visualize Results**: Gain insights from your experiments through intuitive graphs and dashboards.
- **Collaborate**: Share your experiment logs with team members easily, fostering collaboration.
- **Centralize Experiment Management**: Keep a centralized history of all experiments for reproducibility and comparison.

You can sign up for an account [here](https://www.wandb.com/).

### MobaXTerm
To connect to the remote HPC cluster, we'll use MobaXTerm. It provides:
- **SSH Connections**: Securely log in to remote servers.
- **SFTP Browser**: Drag-and-drop file transfers between your local machine and the cluster.
- **Multiple Protocols**: Support for SSH, X11 forwarding, and more, making it a versatile tool for remote work.

Download and installation instructions can be found [here](https://mobaxterm.mobatek.net/).

### Docker (Required for challenge submission)
For challenge submission, your model must be packaged as a self-contained Docker image and exported to a `.tar` file.

- Install **Docker Desktop**: https://www.docker.com/products/docker-desktop/
- During installation, keep default settings unless your system requires specific changes.
- After installation, start Docker Desktop and wait until it reports that Docker is running.

To verify installation, run:

```bash
docker --version
docker run --rm hello-world
```

If these commands work, Docker is ready for building and testing your submission container.

## Basic Usage
### VSCode
- Open VSCode and open your project folder.
- Use the integrated terminal (`Ctrl + Shift + \``) for running scripts and managing Git.
- Install useful extensions:
    - **Python**: Adds support for Python development.
    - **Github Pull Requests**: Simplifies GitHub repository management (recommended).
    - **Remote - SSH**: Enables editing files on the HPC cluister directly from VSCode (optional).

### GitHub
#### Overview
You will work with **two repositories**:
- **Your fork** (where you push your own work)
- **The course repository** (which you pull updates from)

When you **fork and then clone your fork**, GitHub automatically sets this up correctly:
- `origin` → your fork
- `upstream` → the original course repository

This is the **recommended and expected workflow** for this assignment.

#### Step 1: Fork the repository
1. Go to the course repository on GitHub: https://github.com/TUE-VCA/NNCV
2. Click the **Fork** button (top-right corner)
3. Select your own GitHub account

This creates your personal copy of the repository.

#### Step 2: Clone *your fork*
Clone **your fork**, not the original repository:

```bash
git clone https://github.com/<your-username>/NNCV.git
cd NNCV
```

Verify the remotes:

```bash
git remote -v
```

You should see:
- `origin` → your fork
- `upstream` → the original course repository

> If upstream is missing, you can add it manually:
> ```bash
> git remote add upstream https://github.com/TUE-VCA/NNCV.git
> ```

#### Step 3: Working on the assignment
- Make changes locally or on the server
- Commit frequently with meaningful commit messages
- Push only to your fork:

```bash
git commit -m "Commit message"
git push origin main
```

#### Step 4: Pull updates from the course repository
We may push bug fixes or clarifications to the course repository. To stay up to date:

```bash
git fetch upstream
git rebase upstream/main
```

Then push the updated history back to your fork:

```bash
git push --force-with-lease origin main
```

#### Important rules
- ❌ Do **not** push to the course repository
- ❌ Do **not** remove the `upstream` remote
- ✅ Always submit your work via **your fork**

#### HPC Usage and Job Submission
For instructions on cloning your fork on the HPC cluster and submitting jobs, see:
`README-Slurm.md`

### MobaXTerm
- Connect to the remote server:
    - Open MobaXTerm.
    - Click on **Session** > **SSH**.
    - Enter the server details.
        - **Remote host**: snellius.surf.nl.
        - **Specify username**: check box + `<your-username>`.
    - Click **OK**.
 > TIP: Save your sessions in MobaXTerm for faster connections in the future.

### Docker
- Build your submission image from the `Final assignment` folder:
  ```bash
  docker build -t nncv-submission:latest -f "Final assignment/Dockerfile" "Final assignment"
  ```
- Test locally with mounted input/output folders:
  ```bash
  docker run --rm -v "${PWD}/local_data:/data" -v "${PWD}/local_output:/output" nncv-submission:latest
  ```
- Export image as a `.tar` file for upload:
  ```bash
  docker save -o nncv_submission.tar nncv-submission:latest
  ```

For full submission details and challenge server endpoints, see `README-Submission.md`.

## Installation verification checklist
Before starting the assignment, confirm all required tools are available:

- VS Code opens and can open this repository.
- Git is installed:
  ```bash
  git --version
  ```
- Docker is installed and running:
  ```bash
  docker --version
  docker run --rm hello-world
  ```
- You can sign in to GitHub.
- You can sign in to Weights & Biases (W&B).
- MobaXTerm is installed and can start an SSH session.

## Additional tips
### SSH & Authentication
To avoid repeated password prompts:

#### Github
- Create a Personal Access Token:
  Settings → Developer settings → Personal access tokens
- Use it when cloning on the HPC cluster:
  ```bash
  git clone https://<username>:<token>@github.com/<username>/NNCV.git
  ```

#### Weights & Biases
- Create an API key in your W&B account settings
- Usage instructions are provided in `README-Slurm.md`

### Debugging in VS Code
Use breakpoints and the built-in debugger.
Documentation: http://code.visualstudio.com/docs/python/debugging.
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

---

## Basic Usage

### VSCode
- Open VSCode and open your project folder.
- Use the integrated terminal (`Ctrl + Shift + \``) for running scripts and managing Git.
- Install useful extensions:
    - **Python**: Adds support for Python development.
    - **Github Pull Requests**: Simplifies GitHub repository management (recommended).
    - **Remote - SSH**: Enables editing files on the HPC cluister directly from VSCode (optional).

### GitHub
To start working on the project, follow these steps:

#### Step 1: Clone the repository
1. Open a terminal or the VSCode terminal on your local system.
2. Clone this repository to your local machine:  
    ```bash
    git clone https://github.com/TUE-VCA/NNCV
    ```
3. Navigate into the cloned repository directory:
    ```bash
    cd NNCV
    ```

#### Step 2: Create your own repository on GitHub
1. Log in to your GitHub account.
2. Create a new repository. For example, name it `NNCV`. Leave it empty (do not initialize with a README or .gitignore).

#### Step 3: Change the Git Remote to your repository
1. Remove the existing remote link to the original repository:
    ```bash
    git remote remove origin
    ```
2. Add your own repository as the new remote:
    ```bash
    git remote add origin https://github.com/<your-username>/<your-repo-name>
    ```
    Replace `<your-username>` and `<your-repo-name>` with your GitHub username and the name of your new repository.
3. Verify the new remote:
    ```bash
    git remote -v
    ```
    You should see your repository URL listed.

> IMPORTANT NOTE: you can also directly change the current URL instead of first deleting it by using `git remote set-url origin https://github.com/<your-username>/<your-repo-name>` instead of step 1 and 2. This gives you the opportunity to use `git remote set-url --push origin https://github.com/<your-username>/<your-repo-name>` to only change the remote to where we push our code. That way you can still pull from the original directory to have the latest updates.

#### Step 4: Push the code to your repository
1. Push the code from your local machine to your new GitHub repository:
    ```bash
    git push -u origin main
    ```
    Replace `main` with the branch name if it is different.

> For how to pull this repository to the HPC and submit a job to the cluster, check the `README-Slurm.md` file.

### MobaXTerm
- Connect to the remote server:
    - Open MobaXTerm.
    - Click on **Session** > **SSH**.
    - Enter the server details.
        - **Remote host**: snellius.surf.nl.
        - **Specify username**: check box + `<your-username>`.
    - Click **OK**.
 > TIP: Save your sessions in MobaXTerm for faster connections in the future.

## Additional tips
1. **SSH Key Managment**: To avoid typing your password repeatedly when using Git or W&B, set up an API key. 
    - **GitHub**: Go to **Settings** > **Developer settings** > **Personal access tokens** > **Tokens (classic)** and create a personal access token. Then on the server, change your clone command to: 
        ```bash
        git clone https://<your-username>:<your-api-key>@github.com/<your-username>/<your-repo-name>
        ```
        This will make sure that everytime you will perform a `git` operation (e.g., `git pull`) in the future within this repo, your will automatically be logged in.
    - **W&B**: Go to **User settings** > **API keys** and create a new key. You can leave this key as is for now. More instructions on how to use this key are specified in the `README-Slurm.md` file.
2. **Debugging in VSCode**: Use breakpoints and the built-in debugger for easier code debugging. Learn more about debugging in Python [here](http://code.visualstudio.com/docs/python/debugging).
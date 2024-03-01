# Project Initialization Script

## Overview

This Bash script automates the process of initializing a project, creating and activating a Conda environment, installing dependencies, setingup git hooks, git workflow and running tests. It supports various options and configurations to tailor the setup to your project's needs.

## Prerequisites

Before running the script, ensure the following prerequisites are met:
- [Git](https://git-scm.com/) is installed on your system.

## Usage

```bash
./setup.sh [OPTIONS]
```

### Options:

- `--git-url`: Git URL of the project that needs to be initiated.
- `--conda-env-name`: Conda environment name for the project (default is the repository name).
- `--run-test`: Enable or disable running tests after project installation (default is true).
- `--test-mode`: Set the test mode (default is NORMAL, REPORT generates a test report).
- `--dependency-source-paths`: Comma-separated paths of local source dependencies.
- `--help`: Display usage information.

### Example:

```bash
./setup.sh --git-url=https://github.com/your/repository --conda-env-name=myenv --run-test=true --test-mode=NORMAL
```

## Options Details

- **Git URL (`--git-url`):** Specify the Git URL of the project that you want to initialize.

- **Conda Environment Name (`--conda-env-name`):** Set the Conda environment name for the project. If not provided, it defaults to the repository name.

- **Run Test (`--run-test`):** Enable or disable running tests after project installation. The default is true.

- **Test Mode (`--test-mode`):** Set the test mode. The default is NORMAL, but you can set it to REPORT for generating a test report.

- **Dependency Source Paths (`--dependency-source-paths`):** If the project has local source dependencies, provide their paths as a comma-separated list.

- **Help (`--help`):** Display usage information.

## Additional Notes

- If the test mode is set to REPORT, ensure that the `run.sh` script is set up in the project base directory to run tests.

- The script installs Anaconda if not already installed, creates a Conda environment, installs project dependencies, and runs tests.
- Pre-commit is used to settup local git hooks.
- Git hooks are configured using the .pre-commit-config.yaml file provided with the script.

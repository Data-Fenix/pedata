# Github Workflow README

## Github Actions

When cloning the repository using the provided script, Github Actions will be automatically set up for the project. If it's already set up in the repository, the script will skip this step and keep the existing configuration. If you want to forcefully set up Github Actions again, you can delete the `.github` directory and remove it from the git repository. The next time you clone, it will be set up again.

## Ruff Lint Configuration

All the configuration related to Ruff lint is in the `ruff.toml` file in the root directory. You can modify this file to add or ignore rules and make other configurations as needed.

## Unit Test Coverage

The threshold value for unit test coverage can be changed by modifying the `.github/workflows/main.yml` file. Look for the environment variable `MIN_COV` and adjust its value. The job will fail if the test coverage is less than the specified threshold.

## Dependency Resolution

Dependencies are resolved by triggering the provided script `install_Company_dependencies.sh`, and you have the option to enable it by changing the `main.yml` file. Currently, it is disabled.

## Workflow Optimization

Workflows utilize Github caches to reduce execution time. This helps improve the efficiency of the workflows.

## Pull Requests

When creating a pull request, a workflow will run and comment the results on the pull request. This provides visibility into the status of the tests and other checks.
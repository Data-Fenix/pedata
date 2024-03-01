#!/bin/bash
export PATH="$HOME/anaconda3/bin:$PATH" 
# Set default values
git_url=""
conda_env_name=""
repo_name=""
run_test=true
test_mode="NORMAL"
help=false
dependencies_source_paths=""

# ANSI color codes
yellow='\033[1;33m'
reset='\033[0m'

# Function to display an error message and exit
function show_error {
    echo -e "${yellow}Error: $1${reset}"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --git-url=*)
            git_url="${1#*=}"
            ;;
        --conda-env-name=*)
            conda_env_name="${1#*=}"
            ;;
        --run-test=*)
            run_test="${1#*=}"
            ;;
        --test-mode=*)
            test_mode="${1#*=}"
            ;;
        --dependency-source-paths=*)
            dependencies_source_paths="${1#*=}"
            ;;    
        --help*)
            help=true
            ;;
        *)
            show_error "Unknown parameter '$1'"
            ;;
    esac
    shift
done

if $help; then
 echo -e "${yellow}--git-url= git url of the project that need to be initiated. ${reset}"
 echo -e "${yellow}--conda-env-name= the conda enviroment name the project need to be up and running, you can use existing enviroment or provide one if not a new enviroment will be created to the project name. ${reset}"
 echo -e "${yellow}--run-test= this is on by default, and test will be run after the project installation, can be disable using --run-test=false .${reset}"
 echo -e "${yellow}--test-mode= this is normal by default, if you need the test result report you can use. --test-mode=REPORT .${reset}"
 echo -e "${yellow}--dependency-source-paths= if the installing project has local source dependencies that need to be installed need to providide as , --dependency-source-paths=/home/name/Projects/mlops,/home/name/Projects/eep.${reset}"
 exit 1
fi

# Check if git url is provided
[ -z "$git_url" ] && show_error "--git-url= is required."

repo_name=$(basename "$git_url" .git)

# Check if conda env name is provided, otherwise use repo name
[ -z "$conda_env_name" ] && conda_env_name="$repo_name"

# Validate Conda environment name
if [[ ! "$conda_env_name" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*$ ]]; then
    show_error "Invalid Conda environment name. It must start with a letter or underscore and can only contain letters, numbers, underscores, and hyphens."
fi

echo -e "----------git:${git_url}"
echo -e "----------conda-env-name:${conda_env_name}"
echo -e "----------clone-repo:${repo_name}"
echo -e "----------run-test:${run_test}"
echo -e "----------test-mode:${test_mode}"
echo -e "----------dependency-source-paths:${dependencies_source_paths}"


if [[ "$test_mode" == "REPORT" ]]; then
 echo -e "${yellow}You have selcted test-mode as REPORT, have you set up .run.sh to run test in the project base directory ? (Enter YES , NO) ${reset}"
 read answer
 if [[ "$answer" == "NO" ]]; then
    echo -e "${yellow} CANNOT PROCEED !! ${reset}"
    exit 1
 fi
fi 

# Install Anaconda if not installed
if ! command -v conda &> /dev/null; then
    echo -e "${yellow}----------Installing Anaconda${reset}"
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O ~/anaconda.sh  # Replace with the latest version if needed
    bash ~/anaconda.sh -b -p $HOME/anaconda3
    conda update -n base -c defaults conda
    conda config --set auto_activate_base true
    conda init
fi

# Create Conda environment if it doesn't exist
if !(conda env list | grep -q "$conda_env_name"); then
    echo -e "${yellow}----------Create Conda environment: $conda_env_name${reset}"
    conda create -n "$conda_env_name" python=3.11 -y
else
    echo -e "${yellow}Conda environment $conda_env_name already exists.${reset}"
fi

# Activate Conda environment
echo -e "${yellow}----------Activate Conda environment: $conda_env_name${reset}"
source activate $conda_env_name

# Install dependencies
echo -e "${yellow}----------Install project dependencies${reset}"
if command -v pip &> /dev/null; then
  echo "pip is installed."
  pip --version
else
  conda install pip
fi

# Install pytest and dependencies
echo -e "${yellow}----------Install pytest and dependencies${reset}"
pip install pytest pytest-cov pytest-html

# Install dataset
echo -e "${yellow}----------Install Install dataset${reset}"
pip install datasets==2.15.0

# Install pre-commit
echo -e "${yellow}----------Install pre-commit${reset}"
pip install pre-commit

# # Install ruff
# echo -e "${yellow}----------Install ruff${reset}"
# pip install ruff

# Set up pre-commit globally
echo -e "${yellow}----------Setting Up Pre-commit Globally${reset}"
git config --global init.templateDir ~/.git-template
pre-commit init-templatedir ~/.git-template --hook-type pre-commit --hook-type pre-push --no-allow-missing-config


# Clone the repository if not already present
if [ ! -d "$repo_name" ]; then
    echo -e "${yellow}------- Cloning repository: $repo_name${reset}"
    git clone "$git_url"
    echo -e "${yellow}Cloning complete.${reset}"
fi

# install local dependendies
if [ -n "$dependencies_source_paths" ]; then
    IFS=',' read -ra paths <<< "$dependencies_source_paths"
    for path in "${paths[@]}"; do
        echo -e "${yellow}------- installing local dependency: $path${reset}"
        pip install $path
    done
fi


# Install the project
echo -e "${yellow}----------Install the project${reset}"
cd "$repo_name"
pip install .

# create .pre-commit-config.yaml
echo -e "${yellow}----------Creating pre-commit-config.yaml in the project${reset}"
CONFIG_FILE=".pre-commit-config.yaml"

# Check if the file exists, and create or replace accordingly
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating $CONFIG_FILE"
    cp -f ../$CONFIG_FILE .
    git add .pre-commit-config.yaml
    git commit -m "Seting up GitHub pre-commit config" --no-verify
    git push --no-verify
    echo "Done!"
else
    echo "Already Exisits ! $CONFIG_FILE"
fi

# Install git work-flow dependencies
echo -e "${yellow}----------Install git work-flow dependencies ${reset}"
git clone https://github.com/Data-Fenix/gitworkflows.git && sh gitworkflows/setup.sh

# Display current dependency list
echo -e "${yellow}----------Current dependency list${reset}"
conda list

# Run project tests
if $run_test; then
    echo -e "${yellow}----------Run project tests in $test_mode mode ${reset}"
    if [[ "$test_mode" == "REPORT" ]]; then
        if [ -e "run.sh" ]; then
         chmod +x run.sh
         source run.sh
        else
         pytest -s -v test
        fi
    else
      pytest -s -v test
    fi
fi

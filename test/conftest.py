# imports
import subprocess
import shutil
import os
from huggingface_hub import Repository

# import fixtures from pedata.static package
from pedata.static import (
    regr_dataset_train,
    regr_dataset_test,
    regr_dataset,
    regr_dataset_splits,
    regr_dataset_as_1split_dict,
    regr_dna_dataset_train,
    regr_dna_dataset_test,
    regr_dna_dataset,
    regr_dna_dataset_splits,
    regr_dna_dataset_as_1split_dict,
)


def run_command(command):
    """wrapper for subprocess.run to run a command and print the output"""
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print("Command output:", result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print("Command failed. Output:", e.stdout)
        raise


def huggingface_hub_access(
    repo_name="https://huggingface.co/datasets/Exazyme/mock_for_checking_access",
):
    """Check if we are online and can access huggingface"""
    try:
        Repository(
            ".cache/Exazyme/mock_for_checking_access",
            clone_from="Exazyme/mock_for_checking_access",
            repo_type="dataset",
        )
        print(f"Connected to Hugging Face Hub! {repo_name}")
        return True
    except Exception as e:
        print(f"Error accessing Hugging Face Hub: {repo_name}")
        return False


def delete_folder_if_exists(folder_path):
    """Delete a folder and its contents if it exists
    Args:
        folder_path: path to the folder to delete
    Returns:
        True if the folder was there and deleted, False otherwise"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"{folder_path} and its contents deleted successfully.")
        return True
    else:
        print(f"{folder_path} does not exist.")
        return False


def delete_file_if_exists(file_path):
    """Delete a file if it exists
    Args:
        file_path: path to the file to delete
    Returns:
        True if the file was there and deleted, False otherwise"""

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted successfully.")
        return True
    else:
        print(f"{file_path} does not exist.")
        return False

# setting setting the __package__ attribute to solve the relative import proplem when running the scripts in the command line
__package__ = "pedata.hfhub_tools"

# imports
import argparse
import shutil
from pathlib import Path
from ..preprocessing import rename_splits, split_col_histogram
from . import get_dataset, clear_hub_ds_files_and_metadata
import json

def rename_hf_dataset_splits(repo_id: str, split_name_map: dict[str, dict[str, str]] = {}) -> None:
    """Rename the split columns to the new split names

    Args:
        repo_id: The id of the repo to rename the column in
        split_name_map: A dictionary mapping the original split names to the new split names. 
            Example: {"DatasetDict":{"training": "train", "val": "validation"},"random split": {"training": "train", "val": "validation"}, "source": {"testing": "test"}}
            "DatasetDict" is the default key for the DatasetDict object split maps
            Split names similar to train, validation and test will be automatically mapped to the new split names.

    Notes:
        This function pulls from the hub, renames the split column names and pushes it back to the hub

    Raises:
        Exception: If the split columns were not renamed correctly
    """
    # # pull dataset from hub
    cache_root_dir = ".cache"
    dataset = get_dataset(
        repo_id, cache_root_dir=cache_root_dir, delete_cache=False, as_DatasetDict=True
    )
    dataset_histogram = split_col_histogram(dataset)

    dataset, split_name_map = rename_splits(dataset, split_name_map=split_name_map)
    renamed_dataset_histogram = split_col_histogram(dataset)

    for key in dataset_histogram:
        sub_split_name_map = split_name_map.get(key, {})
        assert all(dataset_histogram[key][split]==renamed_dataset_histogram[key][sub_split_name_map.get(split, split)] for split in dataset_histogram[key]), f"The split column {key} was not renamed correctly"

    # clear hub data and metadata
    clear_hub_ds_files_and_metadata(repo_id)

    # push to hub
    dataset.push_to_hub(repo_id, private=True)

    # clear cache
    shutil.rmtree(Path(f"{cache_root_dir}/{repo_id}"), ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename the split columns to the new split names")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="The id of the repo to rename the column in",
        required=True,
    )
    parser.add_argument(
        "--mapping",
        required=False,
        help="JSON string representing a dictionary with old split name as key and new split name as value",
    )
    args = parser.parse_args()

    # parse JSON string to dictionary
    split_name_map = json.loads(args.mapping)
    rename_hf_dataset_splits(args.repo_id, split_name_map=split_name_map)
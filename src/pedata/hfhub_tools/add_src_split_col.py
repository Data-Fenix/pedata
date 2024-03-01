# setting setting the __package__ attribute to solve the relative import proplem when running the scripts in the command line
__package__ = "pedata.hfhub_tools"

# imports
import argparse
import shutil
from pathlib import Path
from ..preprocessing import add_source_split_column, rename_splits, split_col_histogram
from . import get_dataset, clear_hub_ds_files_and_metadata

# method
def add_src_split_col(repo_id: str, merge_split: bool = False, split_name_map: dict[str, dict[str, str]] = {}) -> None:
    """Add the source_split column to the dataset, which correspond to the splits used for the dataset by previous works on the dataset (referenced in the dataset card)

    Args:
        repo_id: The id of the repo to rename the column in
        merge_split: boolean to merge the split columns into one column.
            Defaults to False because it has the least consequences on the structure of the dataset
        split_name_map: A dictionary mapping the original split names to the new split names. 
            Example: {"DatasetDict":{"training": "train", "val": "validation"},"random split": {"training": "train", "val": "validation"}, "source": {"testing": "test"}}
            "DatasetDict" is the default key for the DatasetDict object split maps
            Split names similar to train, validation and test will be automatically mapped to the new split names.
    Notes:
        This function pulls from the hub, renames add the source split column name, optionally merge splits into one (merge_split=True) and pushes it back to the hub

    Raises:
        Exception: If the source split column was not added correctly
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

    # add source split column
    dataset = add_source_split_column(dataset, merge_split=merge_split)
    source_split_histogram = split_col_histogram(dataset)["source_split"]

    # check that the source split column was added and contains the correct keys
    # using an assert because a condition with a raise(Exception) if not testable unless we make a small function for it
    assert all(
        [renamed_dataset_histogram["DatasetDict"][split]==source_split_histogram[split] for split in source_split_histogram]
    ), "The source_split column was not added correctly"


    # clear hub data and metadata
    clear_hub_ds_files_and_metadata(repo_id)

    # push to hub
    dataset.push_to_hub(repo_id, private=True)

    # clear cache
    shutil.rmtree(Path(f"{cache_root_dir}/{repo_id}"), ignore_errors=True)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description=(
            """Add the source_split column to the dataset, which correspond to the splits used for the dataset by previous works on the dataset (referenced in the dataset card)"""
        )
    )
    # required arguments
    parser.add_argument(
        "--repo",
        required=True,
        help="The id of the repo to rename the column in",
    )
    parser.add_argument(
        "--merge_split",
        required=False,
        help="boolean to merge the split columns into one column",
    )
    args = parser.parse_args()

    # # create dataset upload object
    add_src_split_col(repo_id=args.repo, merge_split=args.merge_split)

    # # Example usage:

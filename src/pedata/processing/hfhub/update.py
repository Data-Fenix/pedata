# setting setting the __package__ attribute to solve the relative import proplem when runningn the scripts in the command line
__package__ = "pedata.processing.hfhub"
import argparse
import os
import shutil
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from .base import DatasetHubHandler
from ...hfhub_tools.utils import clear_hub_ds_files_and_metadata


class DatasetUpdate(DatasetHubHandler):
    """A class to handle dataset update and on HuggingFace Hub."""

    def __init__(
        self,
        repo: str,
        commit_hash: str = None,
        local_dir: str | Path = "./local_datasets",
        cache_dir: str | Path = "./cache",
        save_locally: bool = True,
        splits_to_combine_as_whole_ds: list[str] = None,
        needed_encodings: list[str] = None,
    ):
        """Initialize the class and run the creation, update and upload pipeline.
        Args:
            repo (str): Hugging Face Hub repository name (format: 'Exazyme/dataset-name').
            commit_hash: Commit hash of the dataset to pull from Hugging Face. - default: None
            local_dir (str): Local directory to save the dataset to.
            cache_dir (str): cache directory
            save_locally (bool): Whether to save the dataset to a local directory. - default: True
            splits_to_combine_as_whole_ds: The name of the splits to combine as the whole dataset
                - when updating a dataset which is already on the hub. - default: []
            needed_encodings (list): list of encodings for the dataset; default: []

        """
        super().__init__(
            repo,
            commit_hash,
            local_dir,
            cache_dir,
            save_locally,
            splits_to_combine_as_whole_ds,
            needed_encodings,
        )

        self.update_dataset()

    def update_dataset(self) -> None:
        """run the update pipeline"""

        # get the dataset
        self.pull_and_select()

        # preprocess propcess the dataset
        self.prepare_data()

        # clear data and metadata on the remote repository
        self.prepare_repo(verbose=True)

        # clear the cache before pushing
        shutil.rmtree(self._cache_dir, ignore_errors=True)

        # push
        self.push()
        print(self.__repr__())

        # clear the cache again
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def pull_and_select(self) -> None:
        """Pulls the dataset from Hugging Face, update it."""
        dataset_dict = load_dataset(
            f"{self._repo}",
            download_mode="force_redownload",
            cache_dir=Path(self._cache_dir),
            revision=self._commit_hash,
        )
        splits_already_in_dataset = list(dataset_dict.keys())

        if (
            len(dataset_dict) > 1
            and self._whole_split_name not in splits_already_in_dataset
        ):
            raise ValueError(
                f"DatasetDict has more than one split and does not have a split named {self._whole_split_name}."
                "Use splits_to_combine_as_whole_ds as argument to specify which splits to combine."
            )

        # if splits_to_combine_as_whole_ds and there is only one split in the dataset
        if self._splits_to_combine_as_whole_ds == [] and len(dataset_dict) == 1:
            self._splits_to_combine_as_whole_ds = splits_already_in_dataset

        # convert the DatasetDict to a Dataset
        self._dataset = concatenate_datasets(
            [dataset_dict[ds] for ds in self._splits_to_combine_as_whole_ds]
        )

    def prepare_repo(self, verbose: bool = True) -> None:
        """prepare the repo for pushing the dataset to Hugging Face."""
        if verbose:
            print(self.__repr__())

        # create local directory
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)

        # save
        if self._save_locally:
            self._dataset.save_to_disk(self._local_dir)

        # clear the hub dataset files and metadata
        clear_hub_ds_files_and_metadata(self._repo)

    def __repr__(self) -> None:
        """Print the processing to be done or the processing done."""

        def print_list(l):
            return "\n".join([f" - {item}" for item in l])

        if "_dataset" not in self.__dict__:
            return f"""
------------------------------------
DatasetUpdate - Processing to be done
------------------------------------
- repo={self._repo} 
- local_dir={self._local_dir}
- save_locally={self._save_locally}
            """
        else:
            return f"""
-------------------------------
DatasetUpdate - Processing done
-------------------------------
Saved locally: {self.local_path}
Pushed to the huggingface repository: {self._repo}
Available features:
{print_list(self.feature_names)}
Available targets: 
{print_list(self.target_names)}
Available splits:
{print_list(self.available_splits)}
            """


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Create and push a dataset to Hugging Face."
    )
    # required arguments
    parser.add_argument(
        "--repo",
        required=True,
        help="Name of the repository to pull from on Hugging Face.",
    )
    # optional arguments
    parser.add_argument(
        "--commit_hash",
        required=False,
        help="str:The commit hash of the dataset to pull from Hugging Face.",
        default=None,
    )
    parser.add_argument(
        "--local_dir",
        required=False,
        help="Name of the local directory to save the dataset to.",
        default="./local_datasets",
    )
    parser.add_argument(
        "--cache_dir",
        required=False,
        help="cache directory",
        default="./cache",
    )
    parser.add_argument(
        "--save_locally",
        required=False,
        help="Name of the local directory to save the dataset to.",
        default=True,
    )
    parser.add_argument(
        "--splits_to_combine_as_whole_ds",
        required=False,
        help="list: names of the split to combine as 'whole_dataset'",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--needed_encodings",
        required=False,
        help="list: list of encodings for the dataset; default: []",
        nargs="+",
        default=[],
    )

    args = parser.parse_args()
    print(args)
    # create dataset upload object
    DatasetUpdate(
        args.repo,
        commit_hash=args.commit_hash,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        save_locally=args.save_locally,
        splits_to_combine_as_whole_ds=args.splits_to_combine_as_whole_ds,
        needed_encodings=args.needed_encodings,
    )

    # example usage:
    # python examples/dataset_upload.py --repo Exazyme/test_example_dataset_ha1 --filename examples/datasets/test_example_dataset_ha1.csv

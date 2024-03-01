__package__ = "pedata.processing.hfhub"
import argparse
import os
import shutil
from pathlib import Path
import numpy as np
from huggingface_hub import repo_exists
from .base import DatasetHubHandler
from ...data_io.load import load_data


class DatasetUpload(DatasetHubHandler):
    """A class to handle dataset update and on HuggingFace Hub."""

    def __init__(
        self,
        repo: str,
        commit_hash: str = None,
        local_dir: str | Path = "./local_datasets",
        cache_dir: str | Path = "./cache",
        csv_filename: str | Path = None,
        save_locally: bool = True,
        splits_to_combine_as_whole_ds: list = None,
        needed_encodings: list[str] = None,
        overwrite_repo: bool = False,
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

        self._overwrite_repo = overwrite_repo

        # this changes overwrite_repo to False when it has been set to True and the repo does not exists
        if not repo_exists(self._repo, repo_type="dataset") and self._overwrite_repo:
            overwrite_repo = False  # FIXME test this

        if repo_exists(self._repo, repo_type="dataset") and not self._overwrite_repo:
            raise Exception(
                f"repo {self._repo} already exists \n"
                "Please choose another name or set overwrite_repo=True to overwrite the content of the repo"
            )

        self._csv_filename = csv_filename
        self.upload_dataset()

    def upload_dataset(self) -> None:
        """run the uload pipeline"""
        # load the dataset
        self._dataset = load_data(self._csv_filename)

        # preprocess propcess the dataset
        self.prepare_data()

        # clear the cache before pushing
        shutil.rmtree(self._cache_dir, ignore_errors=True)

        # push
        self.push()
        print(self.__repr__())

        # clear the cache again
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def __repr__(self) -> None:
        """Print the processing to be done or the processing done."""

        def print_list(l):
            return "\n".join([f" - {item}" for item in l])

        if "_dataset" not in self.__dict__:
            return f"""
------------------------------------
DatasetLoad - Processing to be done
------------------------------------
- repo={self._repo} 
- file={self._csv_filename}
- local_dir={self._local_dir}
- save_locally={self._save_locally}
            """
        else:
            return f"""
-------------------------------
DatasetLoad - Processing done
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
        "--filename",
        required=False,
        help="Path to the CSV file for dataset creation.",
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

    parser.add_argument(
        "--overwrite_repo",
        required=False,
        help="bool: Set to True to overwrite a repo when uploading a new dataset, if the repo already exists",
        default=False,
    )

    args = parser.parse_args()
    print(args)

    DatasetUpload(
        args.repo,
        commit_hash=args.commit_hash,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        csv_filename=args.filename,
        save_locally=args.save_locally,
        splits_to_combine_as_whole_ds=args.splits_to_combine_as_whole_ds,
        needed_encodings=args.needed_encodings,
        overwrite_repo=args.overwrite_repo,
    )

    # example usage:
    # python examples/dataset_upload.py --repo Exazyme/test_example_dataset_ha1 --filename examples/datasets/test_example_dataset_ha1.csv

__package__ = "pedata.processing.hfhub"
import os
from pathlib import Path
import numpy as np
from typing import Sequence
from ...preprocessing import preprocessing_pipeline
from ...encoding import add_encodings
from ...util import get_target


class DatasetHubHandler:
    """A base class to handle datasets repos on HuggingFace Hub."""

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
            repo: Hugging Face Hub repository name (format: 'Exazyme/dataset-name').
            commit_hash: Commit hash of the dataset to pull from Hugging Face. - default: None
            local_dir: Local directory to save the dataset to.
            cache_dir: cache directory
            save_locally: Whether to save the dataset to a local directory. - default: True
            splits_to_combine_as_whole_ds: The name of the splits to combine as the whole dataset
                - when updating a dataset which is already on the hub. - default: None
            needed_encodings: list of encodings for the dataset; default: None

        """
        self._repo = repo
        self._commit_hash = str(commit_hash) if commit_hash is not None else None
        self._local_dir = Path(local_dir) if isinstance(local_dir, str) else local_dir
        self._cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self._save_locally = save_locally
        self._whole_split_name = "whole_dataset"
        if splits_to_combine_as_whole_ds is None:
            splits_to_combine_as_whole_ds = []
        self._splits_to_combine_as_whole_ds = splits_to_combine_as_whole_ds
        self._needed_encodings = needed_encodings

    def prepare_data(self):
        """Pull a dataset from Hugging Face, update it."""
        # Pull dataset from Hugging Face
        self._dataset = preprocessing_pipeline(
            self._dataset, add_index=True, add_splits=True
        )
        self._dataset = add_encodings(self._dataset, self._needed_encodings)

    def push(self) -> None:
        """Save the dataset to a local directory and push it to Hugging Face."""
        self._dataset.push_to_hub(
            self._repo,
            private=True,
            split="whole_dataset",
            embed_external_files=False,
        )

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

    @property
    def local_path(self) -> str:
        return os.path.join(self._local_dir, self._repo)

    @property
    def figure_path(self) -> str:
        return os.path.join(self.local_path, "figures")

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def target_names(self) -> list[str]:
        """get all target names"""
        targets = get_target(self._dataset, as_dict=True)
        return list(targets.keys())

    @property
    def targets(self) -> dict[Sequence[str], np.ndarray]:
        """Getting all targets
        Returns:
            Dictionary of targets with the target names as keys and the target values as values
        """
        return get_target(self._dataset, as_dict=True)

    @property
    def available_splits(self) -> list[str]:
        """get all available splits"""
        return [col for col in self._dataset.column_names if "split" in col]

    @property
    def feature_names(self) -> list[str]:
        """get all features names"""
        return [
            col
            for col in self._dataset.column_names
            if not (
                col in self.target_names
                or col in self.available_splits
                or col in ["index"]
            )
        ]

    @property
    def datapoints_n(self) -> list[int]:
        """get the number of datapoints"""
        return [self._dataset.num_rows]

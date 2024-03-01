from huggingface_hub import (
    metadata_update,
    list_repo_files,
    delete_file,
    DatasetFilter,
    list_datasets,
)
from datasets import load_dataset, Dataset, concatenate_datasets, DownloadConfig
from pathlib import Path
import shutil


def get_dataset(
    repo_id: str,
    cache_root_dir: str = None,
    use_cache: bool = False,
    delete_cache: bool = False,
    as_DatasetDict: bool = False,
    splits_to_load: list[str] | None = None,
) -> Dataset:
    """
    Get dataset from huggingface hub, optionally using a cache directory, or choosing specific splits and returning a Dataset or a DatasetDict.

    Args:
        dataset_name: The name of the dataset
        cache_root_dir: The root folder for the cache directory to use
        no_cache_dir: Whether to use a cache directory or not
            Default: False: forces to download from hub by deleting the cache directory before loading the dataset
        delete_cache: Whether to delete the cache directory after loading the dataset
        as_DatasetDict: Whether to return the dataset as a DatasetDict or not
            Default: False -> return the dataset as a Dataset
        splits_to_load: List of splits to load - this is useful if you only want to load a subset of the splits which are on the hub

    Returns:
        full_dataset: The full dataset

    Notes:
        This is just a wrapper around load_dataset.
        It uses other methods from the "datasets" package and is quite handy
            - for clearing the cache if needed
            - returning your dataset as a Dataset and not a DatasetDict (which is the default behaviour of load_dataset)

    """
    # configure cache directory
    if cache_root_dir is None:
        cache_root_dir = ".cache"
    repo_cache_dir = Path(f"{cache_root_dir}/{repo_id}")

    if not use_cache:
        # clear cache
        shutil.rmtree(repo_cache_dir, ignore_errors=True)  # delete cache for thge

    # configure download config
    dll_conf = DownloadConfig(cache_dir=repo_cache_dir, force_download=True)

    # pull dataset from hub
    dataset = load_dataset(repo_id, download_config=dll_conf, cache_dir=repo_cache_dir)

    if splits_to_load is None:
        splits_to_load = list(dataset.keys())

    if delete_cache:
        # clear cache
        shutil.rmtree(repo_cache_dir, ignore_errors=True)

    if as_DatasetDict:
        return dataset
    else:
        return concatenate_datasets(
            [dataset[split] for split in dataset if split in splits_to_load]
        )


def check_if_dataset_on_hf(
    dataset_name: str,
) -> bool:
    """Check if a dataset is available on the Exazyme space on HuggingFace

    Args:
        dataset_name: Name of the dataset.

    Returns:
        True if the dataset is available, False otherwise.
    """
    if not isinstance(dataset_name, str):
        raise Warning(
            f"dataset_name is here {type(dataset_name)} but should be a string"
        )

    datasets_exazyme = list_datasets(filter=DatasetFilter(author="Exazyme"))
    list_datasets_exazyme = sorted([dataset.id for dataset in datasets_exazyme])
    for dataset in list_datasets_exazyme:
        if dataset_name == dataset:
            return True
        else:
            continue

    return False


def clear_hub_ds_files_and_metadata(
    repo_id: str, list_of_files_to_delete: list | None = None
) -> None:
    """ "Delete all files in a HuggingFace dataset repository data/ folder as well as the metadata in the dataset card.
    Args:
        repo : Hugging Face Hub repository name (format: 'Exazyme/dataset-name').
        list_of_files_to_delete (list): List of files to delete from the hub.
            default: None -> delete all datafiles not corresponding to the splits present in self.dataset
    """
    # delete all files in the data/ folder
    clear_hub_dataset_files(repo_id, list_of_files_to_delete)

    # update the metadata in the dataset card with nothing in it
    metadata_update(
        repo_id=repo_id,
        metadata={"dataset_info": "nothing in it", "configs": []},
        repo_type="dataset",
        overwrite=True,
    )


def clear_hub_dataset_files(
    repo_id, list_of_files_to_delete: list | None = None
) -> None:
    """Delete all files in a HuggingFace dataset repository data/ folder.
    Args:
        dataset (ds.Dataset): Dataset to save and push.
        list_of_files_to_delete (list): List of files to delete from the hub.
            default: [] -> delete all datafiles not corresponding to the splits present in self.dataset
    """
    if list_of_files_to_delete is None:
        list_of_files_to_delete = repo_get_file_list_split_cleanup(repo_id)

    if list_of_files_to_delete is not None:
        print(f"{list_of_files_to_delete} will be deleted from {repo_id}")

        for file_to_delete in list_of_files_to_delete:
            print(f"Deleting {file_to_delete}")
            delete_file(
                path_in_repo=file_to_delete,
                repo_id=repo_id,
                repo_type="dataset",
            )

    else:
        print(f"No dataset files to delete from {repo_id}")


def repo_get_file_list_split_cleanup(repo_id: str) -> list[str]:
    """Delete all files in a HuggingFace dataset repository.
    Checks the difference between the list of files in the repo and the list of files in the dataset directory using the load_dataset methods.
    Args:
        repo_id : Hugging Face Hub repository name (format: 'Exazyme/dataset-name').
    """
    dataset_files = get_hub_data_folder_files(repo_id)
    if dataset_files == []:
        return []

    dataset_dict_keys = list(load_dataset(f"{repo_id}").keys())
    list_of_files_to_delete = []
    for dataset_dict_key in dataset_dict_keys:
        for dataset_file in dataset_files:
            if dataset_dict_key in dataset_file:
                list_of_files_to_delete.append(dataset_file)

    return list_of_files_to_delete


def get_hub_data_folder_files(repo_id: str) -> list[str]:
    """Get all files in a HuggingFace dataset repository data/ folder.
    Args:
        repo_id : Hugging Face Hub repository name (format: 'Exazyme/dataset-name').
    Returns:
        list: List of files in the data/ folder of the repository.
    """
    return [
        file for file in get_all_files_in_ds_repo(repo_id=repo_id) if "data/" in file
    ]


def get_all_files_in_ds_repo(repo_id: str) -> list[str]:
    """Get all files in a HuggingFace dataset repository folder.
    Args:
        repo_id : Hugging Face Hub repository name (format: 'Exazyme/dataset-name').
    Returns:
        list: List of files in the repository.
    """
    return list_repo_files(repo_id=repo_id, repo_type="dataset")

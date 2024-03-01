import os
import subprocess
import shutil

# dataset
from huggingface_hub import delete_repo, repo_exists
from datasets import load_dataset

# pedata
from pedata.processing.hfhub import DatasetUpdate, DatasetUpload
from pedata.data_io import save_dataset_as_csv
from .conftest import (
    huggingface_hub_access,
    run_command,
    delete_folder_if_exists,
    delete_file_if_exists,
)

# pytest
from pytest import fixture, raises

if huggingface_hub_access():

    def delete_cache_and_repo(cache_dir: str, repo_name: str) -> None:
        """Delete the cache and the repo
        Args:
            cache_dir: cache dir to delete
            repo_name: repo name to delete
        """
        shutil.rmtree(cache_dir, ignore_errors=True)
        if repo_exists(repo_id=repo_name, repo_type="dataset"):
            delete_repo(repo_id=repo_name, repo_type="dataset")

    # ========== FIXTURES ==========
    @fixture(scope="module")
    def repo_name():
        return "Exazyme/TestDataset_upload_update_delete"

    @fixture(scope="module")
    def cache_dir():
        return "./cache"

    @fixture(scope="module")
    def csv_filename():
        return "TestDataset_upload_update_delete.csv"

    @fixture(scope="module")
    def needed_encodings():
        return ["aa_seq", "aa_1hot"]

    @fixture(scope="module")
    def updated_encodings():
        return ["aa_unirep_1900"]

    def test_Dataset_upload_and_update(
        regr_dataset,
        needed_encodings,
        csv_filename,
        updated_encodings,
        repo_name,
        cache_dir,
    ):

        # cleanup in case test failed before
        delete_cache_and_repo(cache_dir, repo_name)

        # create a csv file containing the dataset
        save_dataset_as_csv(regr_dataset, csv_filename)

        # 1 ==== uploading the dataset
        DatasetUpload(
            repo=repo_name,
            save_locally=False,
            csv_filename=csv_filename,
            needed_encodings=needed_encodings,
            cache_dir=cache_dir,
        )
        # load the dataset from huggingface and check that it has the correct encodings
        dll_ds = load_dataset(
            repo_name, download_mode="force_redownload", cache_dir=cache_dir
        )
        assert all(
            [
                feature
                in {
                    "aa_mut",
                    "target_kcat_per_kmol",
                    "aa_seq",
                    "aa_1hot",
                    "index",
                    "random_split_train_0_8_test_0_2",
                    "random_split_10_fold",
                }
                for feature in dll_ds["whole_dataset"].info.features
            ]
        )

        # cleanup
        os.remove(csv_filename)  # csvfile
        shutil.rmtree(cache_dir)  # cache dir

        # 2 ==== Update the dataset with the encodings per default
        DatasetUpdate(
            repo=repo_name, needed_encodings=updated_encodings, cache_dir=cache_dir
        )

        # load the dataset from huggingface and check that it has the correct encodings
        dll_ds = load_dataset(
            repo_name, download_mode="force_redownload", cache_dir=cache_dir
        )

        assert all(
            [
                feature
                in {
                    "aa_mut",
                    "target_kcat_per_kmol",
                    "aa_seq",
                    "aa_1hot",
                    "index",
                    "random_split_train_0_8_test_0_2",
                    "random_split_10_fold",
                    "aa_unirep_1900",
                    "aa_unirep_final",
                }
                for feature in dll_ds["whole_dataset"].info.features
            ]
        )

        ## delete the cache
        shutil.rmtree(cache_dir)

        # 4 ======== Specifying splits to combine as a whole dataset
        DatasetUpdate(
            repo=repo_name,
            splits_to_combine_as_whole_ds=["whole_dataset"],
        )

        assert all(
            [
                feature
                in {
                    "aa_mut",
                    "target_kcat_per_kmol",
                    "aa_seq",
                    "aa_1hot",
                    "index",
                    "random_split_train_0_8_test_0_2",
                    "random_split_10_fold",
                    "aa_unirep_1900",
                    "aa_unirep_final",
                }
                for feature in dll_ds["whole_dataset"].info.features
            ]
        )

        # delete the cache and repo
        delete_cache_and_repo(cache_dir, repo_name)

    # =======  These tests are testing additional features and also test running in the command line

    def test_upload_csv_to_huggingface_repo_already_exists_cli():
        """trying to upload to a repo which already exist"""

        # raising error because repo already exists
        wrong_command = (
            "python "
            "src/pedata/processing/hfhub/upload.py "
            "--repo Exazyme/test_example_dataset_ha1 "
            "--filename local_datasets/datafiles/example_dataset_ha1.csv "  # optional
            "--needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900' "  # optional
        )
        with raises(Exception):
            run_command(wrong_command)

        # forcing overwrite
        command = (
            "python "
            "src/pedata/processing/hfhub/upload.py "
            "--repo Exazyme/test_example_dataset_ha1_2 "
            "--filename local_datasets/datafiles/example_dataset_ha1.csv "  # optional
            "--needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900' "  # optional
            "--overwrite_repo True "
        )
        run_command(command)

    def test_update_huggingface_repo_cli():
        # pulling a dataset from huggingface, updating it and uploading it uploading to the same repo"""

        command = (
            "python "
            "src/pedata/processing/hfhub/update.py "
            "--repo Exazyme/test_example_dataset_ha1 "
            "--needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900' "
        )
        run_command(command)

        [
            delete_file_if_exists(file)
            for file in [
                "local_datasets/data-00000-of-00001.arrow",
                "local_datasets/dataset_info.json",
                "local_datasets/state.json",
            ]
        ]

# imports
from datasets import Dataset, DatasetDict
from pytest import fixture
import shutil
from datasets import load_dataset, DownloadConfig

# add_src_split_col
from pedata.hfhub_tools import add_src_split_col

# utils.py
from pedata.hfhub_tools import (
    clear_hub_ds_files_and_metadata,
    get_dataset,
    check_if_dataset_on_hf,
    rename_hub_dataset_column,
)

# conftest
from .conftest import run_command, huggingface_hub_access

# == helper functions ==

if huggingface_hub_access():

    def create_test_repo(repo):
        # forcing overwrite
        command = (
            "python "
            "src/pedata/processing/hfhub/upload.py "
            f"--repo {repo} "
            "--filename local_datasets/datafiles/example_dataset_ha1.csv "  # optional
            "--needed_encodings 'aa_seq' "  # optional
            "--overwrite_repo True "
        )
        run_command(command)

    @fixture()
    def source_split_test_repo():
        return "Exazyme/test_add_source_split_col"

    @fixture()
    def rename_splits_test_repo():
        return "Exazyme/test_rename_splits"

    def test_check_if_dataset_on_hf(source_split_test_repo):
        # check if the function finds one of the datasets
        assert check_if_dataset_on_hf(source_split_test_repo)

    def test_check_if_fake_dataset_on_hf():
        # check if the function does not find a dataset that is not there
        assert not check_if_dataset_on_hf("Exazyme/CrHydA1_regr_fake")

    # == add_src_split_col ==
    def test_add_src_split_col(source_split_test_repo):
        # reset the test repo
        create_test_repo(repo=source_split_test_repo)
        # dataset = get_dataset(
        #     repo_id="Exazyme/test_add_source_split_col",
        #     delete_cache=True,
        #     as_DatasetDict=True,
        # )
        # source_keys = dataset.column_names.keys()
        # add the source split column
        add_src_split_col(repo_id=source_split_test_repo, merge_split=True)

        dataset = get_dataset(
            repo_id=source_split_test_repo,
            delete_cache=True,
            as_DatasetDict=False,
        )
        assert "source_split" in dataset.features.keys()

    def test_add_src_split_col_cli(source_split_test_repo):
        # reset the test repo
        create_test_repo(repo=source_split_test_repo)

        # cli command
        command = (
            "python src/pedata/hfhub_tools/add_src_split_col.py "
            f"--repo {source_split_test_repo} "
            "--merge_split True "
        )

        # run command
        run_command(command)

        # assertion
        dataset = get_dataset(
            repo_id=source_split_test_repo,
            delete_cache=True,
            as_DatasetDict=False,
        )
        assert "source_split" in dataset.features.keys()

    # == clear_hub_ds_files_and_metadata ==

    @fixture()
    def clear_hub_data_test_repo():
        return "Exazyme/test_repo_clear_hub_data"

    def test_clear_hub_ds_files_and_metadata(clear_hub_data_test_repo):
        # reset the test repo
        create_test_repo(repo=clear_hub_data_test_repo)

        # test
        clear_hub_ds_files_and_metadata(repo_id=clear_hub_data_test_repo)
        assert True

    # == get_dataset ==

    @fixture()
    def get_dataset_test_repo():
        return "Exazyme/test_example_dataset_ha1"

    def test_get_dataset(get_dataset_test_repo):
        # returns DatasetDict, NOT using cache and not deleting cache

        dataset = get_dataset(
            repo_id=get_dataset_test_repo,
            delete_cache=False,
            as_DatasetDict=True,
        )
        assert isinstance(dataset, DatasetDict)

        # returns Dataset, selecting splits, using cache and not deleting cache
        dataset = get_dataset(
            repo_id=get_dataset_test_repo,
            use_cache=True,
            delete_cache=False,
            splits_to_load=["whole_dataset"],
        )
        assert isinstance(dataset, Dataset)

        # returns DatasetDict, using cache and deleting cache
        dataset = get_dataset(
            repo_id=get_dataset_test_repo,
            as_DatasetDict=True,
            use_cache=True,
            delete_cache=True,
        )
        assert isinstance(dataset, DatasetDict)

    ## === test rename splits ====

    def test_rename_splits(rename_splits_test_repo):
        create_test_repo(repo=rename_splits_test_repo)
        command = (
            "python src/pedata/hfhub_tools/rename_splits.py "
            f"--repo {rename_splits_test_repo} "
            "--mapping {\} "
        )

        # run command
        run_command(command)
        assert True

    ## === test test_rename_hub_dataset_column ====

    def test_rename_hub_dataset_column():
        """testing the rename_hub_dataset_column function"""

        # helper functions
        def cleanup():
            shutil.rmtree(cache_directory, ignore_errors=True)

        def get_feat_list():
            cleanup()
            inf = load_dataset(
                "Exazyme/test_example_dataset_ha1",
                download_config=dll_conf,
                download_mode="force_redownload",
                cache_dir=cache_directory,
            )
            cleanup()
            return list(inf[list(inf.keys())[0]].features)

        # clear cache
        cache_directory = "./cachebleeeeh"
        dll_conf = DownloadConfig(cache_dir=cache_directory, force_download=True)
        cleanup()

        ## creates a repo with the original names
        command = (
            "python "
            "src/pedata/processing/hfhub/upload.py "
            "--repo Exazyme/test_example_dataset_ha1 "
            "--filename local_datasets/datafiles/example_dataset_ha1.csv "  # optional
            "--needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900' "  # optional
            "--overwrite_repo True "
        )
        run_command(command)

        # changing the column name
        rename_hub_dataset_column(
            repo_id="Exazyme/test_example_dataset_ha1",
            column_name_mapping={
                "aa_mut": "bleeeeh",
            },
        )

        # clear cache and get features list
        list_feat = get_feat_list()
        assert "bleeeeh" in list_feat
        assert "aa_mut" not in list_feat

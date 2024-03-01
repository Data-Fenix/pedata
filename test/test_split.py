import datasets
import numpy as np
import pytest
from pytest import fixture
from pedata.preprocessing import (
    DatasetSplitterRandomTrainTest,
    DatasetSplitterRandomKFold,
    add_source_split_column,
    append_split_columns_to_dataset,
    ZeroShotSpliter,
    get_split_col_names,
    get_split_map,
    rename_splits,
    split_col_histogram,
)


# toy dataset
t_dataset = datasets.Dataset.from_dict(
    {
        "aa_seq": [
            "MLGLYITR",
            "MAGLYITR",
            "MLYLYITR",
            "RAGLYITR",
            "MLRLYITR",
            "RLGLYITR",
        ],
        "target a": [1, 2, 3, 5, 4, 6],
        "aa_mut": [
            "A2L",
            "wildtype",
            "A2L_G3Y",
            "M1R",
            "A2L_G3R",
            "A2L_M1R",
        ],
    }
)

# toy zero shot dataset
t_dataset_zero_shot = datasets.Dataset.from_dict(
    {
        "aa_seq": [
            "MLGLYITR",
            "MAGLYITR",
            "MLYLYITR",
            "RAGLYITR",
            "MLRLYITR",
            "RLGLYITR",
        ],
        "target a": [1, 2, 3, 5, 4, 6],
        "aa_mut": [
            "A2L",
            "wildtype",
            "A2L_G3Y",
            "M1R",
            "A2L_G3R",
            "A2L_M1R",
        ],
        "source_split": ["train", "train", "train", "validation", "validation", "test"],
    }
)

t_zero_shot_train = datasets.Dataset.from_dict(
    {
        "aa_seq": [
            "MLGLYITR",
            "MAGLYITR",
            "MLYLYITR",
        ],
        "target a": [1, 2, 3],
        "aa_mut": [
            "A2L",
            "wildtype",
            "A2L_G3Y",
        ],
        "source_split": ["train", "train", "train"],
    }
)
t_zero_shot_validation = datasets.Dataset.from_dict(
    {
        "aa_seq": [
            "RAGLYITR",
            "MLRLYITR",
        ],
        "target a": [5, 4],
        "aa_mut": [
            "M1R",
            "A2L_G3R",
        ],
        "source_split": ["validation", "validation"],
    }
)
t_zero_shot_test = datasets.Dataset.from_dict(
    {
        "aa_seq": [
            "RLGLYITR",
        ],
        "target a": [6],
        "aa_mut": [
            "A2L_M1R",
        ],
        "source_split": ["test"],
    }
)

t_zero_shot_splitted = datasets.DatasetDict(
    {
        "train": t_zero_shot_train,
        "validation": t_zero_shot_validation,
        "test": t_zero_shot_test,
    }
)

t_dataset_with_split_cols = datasets.Dataset.from_dict(
    {
        "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR"],
        "target a": [1, 2, 3, 5, 4],
        "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R"],
        "split_random": ['training', 'training', 'training', 'testing', 'testing'],
        "splittraintest": ['training', 'training', 'training', 'testing', 'testing'],
        "RandomSplit": ['training', 'training', 'training', 'testing', 'testing'],
        "source split": ['training', 'training', 'training', 'testing', 'testing'],
        "split_k_fold": ['train', 'train', 'train', 'test', 'test'],
        "split-k-fold": ['train', 'train', 'train', 'test', 'test'],
    }
)

@fixture(scope="function")
def toy_dataset_with_split_cols_as_DatasetDict() -> datasets.DatasetDict:
    """Return a toy dataset as a DatasetDict"""
    return datasets.DatasetDict({"training": t_dataset_with_split_cols, "testing": t_dataset_with_split_cols, "validation": t_dataset_with_split_cols})

@fixture(scope="function")
def toy_zero_shot_splitted() -> datasets.DatasetDict:
    """Return a toy dataset"""
    return t_zero_shot_splitted


@fixture(scope="function")
def toy_dataset_zero_shot_as_DatasetDict() -> datasets.DatasetDict:
    """Return a toy dataset as a DatasetDict"""
    return datasets.DatasetDict({"all": t_dataset_zero_shot})


@fixture(scope="function")
def toy_dataset_zero_shot_wo_source_column_as_DatasetDict() -> datasets.DatasetDict:
    """Return a toy dataset as a DatasetDict"""
    return datasets.DatasetDict(
        {"all": t_dataset_zero_shot.remove_columns("source_split")}
    )


# FIXME use the conftest dataset_test once merged add (scope="session") to the fixture - so the dataset does not get remade for each test
@fixture(scope="function")
def toy_dataset() -> datasets.Dataset:
    """Return a toy dataset"""
    return t_dataset


@fixture(scope="function")
def toy_dataset_as_DatasetDict() -> datasets.DatasetDict:
    """Return a toy dataset as a DatasetDict"""
    return datasets.DatasetDict({"all": t_dataset})


def test_dataset_splitter_returns_Dataset(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class returns a Dataset when return_dataset_dict=False"""
    train_test_ds = DatasetSplitterRandomTrainTest().split(
        dataset=toy_dataset,
        return_dataset_dict=False,
        rng_seed=0,
    )

    assert isinstance(train_test_ds, datasets.Dataset)


def test_dataset_splitter_returns_DatasetDict(toy_dataset, toy_dataset_as_DatasetDict):
    """Test that the DatasetSplitterRandomTrainTest class returns a DatasetDict when return_dataset_dict=True"""
    # with Dataset as input
    train_test_ds = DatasetSplitterRandomTrainTest().split(
        dataset=toy_dataset,
        return_dataset_dict=True,
        rng_seed=0,
    )

    assert isinstance(train_test_ds, datasets.DatasetDict)

    # with DatasetDict as input
    train_test_ds = DatasetSplitterRandomTrainTest().split(
        dataset=toy_dataset_as_DatasetDict,
        return_dataset_dict=True,
        rng_seed=0,
    )
    assert isinstance(train_test_ds, datasets.DatasetDict)


def test_dataset_splitter_random_train_tes_default_seed_reprod(
    toy_dataset_as_DatasetDict,
):
    """Test that withthe default random seed, the split is always the same when the operation is repeated twice"""
    for _ in range(2):
        train_test_ds = DatasetSplitterRandomTrainTest().split(
            dataset=toy_dataset_as_DatasetDict,
            return_dataset_dict=True,
        )

        assert isinstance(train_test_ds, datasets.DatasetDict)

        train_test_ds["test"]["aa_seq"]

        assert train_test_ds["test"]["aa_seq"] == [
            "RAGLYITR",
            "MLYLYITR",
        ], f"default rng_seed; train_test_ds['test']['aa_seq'] = {train_test_ds['test']['aa_seq']} but should be ['RLGLYITR', 'MLRLYITR']"


def test_dataset_splitter_random_train_tes_split_diff_seed(toy_dataset_as_DatasetDict):
    """Test that the DatasetSplitterRandomTrainTest class returns the same DatasetDict when the seed is set
    Test that setting the seed with a given value always returns the same DatasetDict as output
    """
    train_test_ds = DatasetSplitterRandomTrainTest().split(
        dataset=toy_dataset_as_DatasetDict,
        return_dataset_dict=True,
        rng_seed=12053,
    )

    assert isinstance(train_test_ds, datasets.DatasetDict)

    train_test_ds["test"]["aa_seq"]
    print(train_test_ds["test"]["aa_seq"])
    assert train_test_ds["test"]["aa_seq"] == [
        "MLYLYITR",
        "MLRLYITR",
    ], f"rng_seed= 12053; train_test_ds['test']['aa_seq'] = {train_test_ds['test']['aa_seq']} but should be ['RLGLYITR', 'MLRLYITR']"


def test_dataset_splitter_concatenating_only_one_split(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class returns the same DatasetDict when the seed is set
    Test that setting the seed with a given value always returns the same DatasetDict as output
    """
    splitter = DatasetSplitterRandomTrainTest()
    dataset = splitter.split(
        dataset=toy_dataset,
        return_dataset_dict=True,
    )
    assert isinstance(dataset, datasets.DatasetDict)
    dataset_cat = splitter.concatenated_dataset(dataset, split_list=["train"])
    assert dataset_cat["aa_seq"] == ["RLGLYITR", "MLRLYITR", "MLGLYITR", "MAGLYITR"]


def test_input_improper_datasetdict(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class returns the same DatasetDict when the seed is set
    Test that setting the seed with a given value always returns the same DatasetDict as output
    """
    splitter = DatasetSplitterRandomTrainTest()
    dataset_dict = splitter.split(
        dataset=toy_dataset,
        return_dataset_dict=True,
    )
    with pytest.raises(ValueError):
        splitter = DatasetSplitterRandomTrainTest().split(
            dataset=dataset_dict, rng_seed=0
        )


def test_dataset_splitter_k_fold(toy_dataset):
    """test that the DatasetSplitterRandomKFold class returns the correct number of splits"""
    splitter = DatasetSplitterRandomKFold(k=3)
    dataset = splitter.split(
        toy_dataset,
        return_dataset_dict=True,
    )

    assert len(dataset) == 3
    assert dataset["split_0"]["aa_seq"] == ["RAGLYITR", "MLYLYITR"]


def test_dataset_splitter_k_fold_splits_are_already_there(toy_dataset):
    """test that the DatasetSplitterRandomKFold class returns the correct number of splits"""
    splitter = DatasetSplitterRandomKFold(k=3)
    dataset = splitter.split(
        dataset=toy_dataset,
        return_dataset_dict=False,
    )

    splitter2 = DatasetSplitterRandomKFold(k=3)
    dataset2 = splitter2.split(
        dataset=toy_dataset,
        return_dataset_dict=False,
    )
    assert (
        dataset2["target a"] == dataset["target a"]
        and dataset2["aa_seq"] == dataset["aa_seq"]
    )


def test_dataset_splitter_k_fold_yielf_train_tes_sets(toy_dataset):
    """Test that the DatasetSplitterRandomKFold class yields the correct train test sets"""
    splitter = DatasetSplitterRandomKFold(k=3)
    dataset = splitter.split(
        dataset=toy_dataset,
    )
    k = 0
    for train_test_set in splitter.yield_all_train_test_sets(dataset):
        if k == 0:
            assert train_test_set["train"]["aa_seq"] == [
                "RLGLYITR",
                "MLRLYITR",
                "MLGLYITR",
                "MAGLYITR",
            ]
        elif k == 1:
            assert train_test_set["train"]["aa_seq"] == [
                "RAGLYITR",
                "MLYLYITR",
                "MLGLYITR",
                "MAGLYITR",
            ]
        k += 1


def test_dataset_splitter_k_fold_yielf_train_tes_sets_2(toy_dataset):
    """Test that the DatasetSplitterRandomKFold class yields the correct train test sets"""
    splitter = DatasetSplitterRandomTrainTest()
    dataset = splitter.split(
        dataset=toy_dataset,
    )
    k = 0
    for train_test_set in splitter.yield_all_train_test_sets(dataset):
        assert train_test_set["train"]["aa_seq"] == [
            "RLGLYITR",
            "MLRLYITR",
            "MLGLYITR",
            "MAGLYITR",
        ]
        k += 1

    assert k == 1


def test_dataset_splitter_k_fold_yielf_train_tes_sets_combined_n(toy_dataset):
    """Test that the DatasetSplitterRandomKFold class yields the correct train test sets when combined_n=2"""
    splitter = DatasetSplitterRandomKFold(k=6)
    dataset = splitter.split(
        dataset=toy_dataset,
    )

    for split_n, split in enumerate(
        splitter.yield_all_train_test_sets(dataset, combined_n=2)
    ):
        if split_n == 0:
            assert split["test"]["aa_seq"][:2] == ["RAGLYITR", "MLYLYITR"]
        elif split_n == 1:
            assert split["train"]["aa_seq"][:2] == ["MLYLYITR", "MLRLYITR"]


def test_append_split_columns_to_dataset(toy_dataset_as_DatasetDict):
    _ = append_split_columns_to_dataset(
        dataset=toy_dataset_as_DatasetDict,
    )


def test_add_source_split_column(toy_dataset_as_DatasetDict):
    dataset = add_source_split_column(toy_dataset_as_DatasetDict)
    assert "source_split" in list(dataset["all"].features)

    dataset_2 = add_source_split_column(dataset, merge_split=True)
    assert all(
        [
            "all" in split_name
            for split_name in dataset_2["whole_dataset"]["source_split"]
        ]
    )


def test_zero_shot_splitter(
    toy_dataset_zero_shot_as_DatasetDict, toy_zero_shot_splitted
):
    """
    Test the zero shot splitter, when the source_split column is present.
    """
    splitter = ZeroShotSpliter("source_split")
    dataset = splitter.split(dataset=toy_dataset_zero_shot_as_DatasetDict, rng_seed=0)
    assert dataset["train"]["aa_seq"] == toy_zero_shot_splitted["train"]["aa_seq"]
    assert (
        dataset["validation"]["aa_seq"]
        == toy_zero_shot_splitted["validation"]["aa_seq"]
    )
    assert dataset["test"]["aa_seq"] == toy_zero_shot_splitted["test"]["aa_seq"]


def test_zero_shot_splitter_without_source_split_column(
    toy_dataset_zero_shot_wo_source_column_as_DatasetDict,
):
    """
    Test the zero shot splitter, when the source_split column is not present.
    """
    splitter = ZeroShotSpliter("source_split")
    with pytest.raises(ValueError):
        _ = splitter.split(
            dataset=toy_dataset_zero_shot_wo_source_column_as_DatasetDict,
        )

def test_get_split_col_names(toy_dataset_with_split_cols_as_DatasetDict):
    split_col_names = get_split_col_names(toy_dataset_with_split_cols_as_DatasetDict)
    assert split_col_names == ['split_random', 'splittraintest', 'RandomSplit', 'source split', 'split_k_fold', 'split-k-fold']

def test_get_split_map():
    splits = ['training', 'val', 'testing', 'keep']
    split_name_map = {'keep': 'kept out'}
    split_map = get_split_map(splits, split_name_map)
    expected_map = {'training': 'train', 'val': 'validation', 'testing': 'test', 'keep': 'kept out'}
    assert split_map == expected_map

def test_rename_splits(toy_dataset_with_split_cols_as_DatasetDict):
    dataset, split_name_map = rename_splits(toy_dataset_with_split_cols_as_DatasetDict)
    assert dataset['train']['split_random'] == ['train', 'train', 'train', 'test', 'test']
    assert dataset['train']['splittraintest'] == ['train', 'train', 'train', 'test', 'test']
    assert dataset['validation']['RandomSplit'] == ['train', 'train', 'train', 'test', 'test']
    assert dataset['validation']['source split'] == ['train', 'train', 'train', 'test', 'test']
    assert dataset['test']['split_k_fold'] == ['train', 'train', 'train', 'test', 'test']
    assert dataset['test']['split-k-fold'] == ['train', 'train', 'train', 'test', 'test']

def test_split_col_histogram(toy_dataset_with_split_cols_as_DatasetDict):
    histogram = split_col_histogram(toy_dataset_with_split_cols_as_DatasetDict)
    assert histogram == {'DatasetDict': {'testing': 5, 'training': 5, 'validation': 5},'split_random': {'training': 9, 'testing': 6}, 'splittraintest': {'training': 9, 'testing': 6}, 'RandomSplit': {'training': 9, 'testing': 6}, 'source split': {'training': 9, 'testing': 6}, 'split_k_fold': {'train': 9, 'test': 6}, 'split-k-fold': {'train': 9, 'test': 6}}
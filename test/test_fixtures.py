# datasets
from datasets import Dataset, DatasetDict

# pytest
from pytest import fixture


@fixture(scope="module")
def needed_encodings():
    return ["aa_seq"]


# testing the fixtures and check that they remain the same to ensure realiable testing
def test_regr_dataset_train(regr_dataset_train):
    assert isinstance(regr_dataset_train, Dataset)
    assert len(regr_dataset_train) == 16


def test_regr_dataset_test(regr_dataset_test):
    assert isinstance(regr_dataset_test, Dataset)
    assert len(regr_dataset_test) == 5


def test_regr_dataset(regr_dataset):
    assert isinstance(regr_dataset, Dataset)
    assert len(regr_dataset) == 21


def test_regr_dataset_splits(regr_dataset_splits):
    assert isinstance(regr_dataset_splits, DatasetDict)
    assert len(regr_dataset_splits) == 2


def test_regr_dataset_as_1split_dict(regr_dataset_as_1split_dict):
    assert isinstance(regr_dataset_as_1split_dict, DatasetDict)
    assert len(regr_dataset_as_1split_dict) == 1


@fixture(scope="module")
def needed_dna_encodings():
    return ["dna_seq"]


def test_regr_dna_dataset_train(regr_dna_dataset_train):
    assert isinstance(regr_dna_dataset_train, Dataset)
    assert len(regr_dna_dataset_train) == 18


def test_regr_dna_dataset_test(regr_dna_dataset_test):
    assert isinstance(regr_dna_dataset_test, Dataset)
    assert len(regr_dna_dataset_test) == 5


def test_regr_dna_dataset(regr_dna_dataset):
    assert isinstance(regr_dna_dataset, Dataset)
    assert len(regr_dna_dataset) == 23


def test_regr_dna_dataset_splits(regr_dna_dataset_splits):
    assert isinstance(regr_dna_dataset_splits, DatasetDict)
    assert len(regr_dna_dataset_splits) == 2


def test_regr_dna_dataset_as_1split_dict(regr_dna_dataset_as_1split_dict):
    assert isinstance(regr_dna_dataset_as_1split_dict, DatasetDict)
    assert len(regr_dna_dataset_as_1split_dict) == 1

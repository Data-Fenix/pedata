from pytest import fixture
from .data import RegressionToyDataset, RegressionDNAToyDataset
from datasets import Dataset, DatasetDict

# TODO: move those fixtures in the peadata repo so they can be imported by eep


@fixture(scope="module")
def needed_dna_encodings():
    return ["dna_seq"]


@fixture(scope="module")
def needed_encodings():
    return ["aa_seq"]


@fixture(scope="module")
def regr_dataset_train(needed_encodings: list[str]) -> Dataset:
    """Regression dataset - train split
    Args:
        needed_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset_train):
        ... "your function here"
        regr_dataset_train will use needed_encodings
    """
    ds = RegressionToyDataset(needed_encodings)
    return ds.train


@fixture(scope="module")
def regr_dataset_test(needed_encodings: list[str]) -> Dataset:
    """Regression dataset - test split
    Args:
        needed_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset_test):
        ... "your function here"
        regr_dataset_test will use needed_encodings
    """
    ds = RegressionToyDataset(needed_encodings)
    return ds.test


@fixture(scope="module")
def regr_dataset(needed_encodings: list[str]) -> Dataset:
    """Regression dataset - full dataset
    Args:
        needed_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset):
        ... "your function here"
        regr_dataset will use needed_encodings
    """

    ds = RegressionToyDataset(needed_encodings)
    return ds.full_dataset


@fixture(scope="module")
def regr_dataset_splits(needed_encodings) -> Dataset:
    """Regression dataset - full dataset
    Args:
        needed_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset_splits):
        ... "your function here"
        regr_dataset_splits will use needed_encodings
    """
    ds = RegressionToyDataset(needed_encodings)
    return ds.train_test_split_dataset


@fixture(scope="module")
def regr_dataset_as_1split_dict(needed_encodings) -> Dataset:
    """Regression dataset - full dataset
    Args:
        needed_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset_splits):
        ... "your function here"
        regr_dataset_splits will use needed_encodings
    """
    ds = RegressionToyDataset(needed_encodings)
    return DatasetDict({"whole_dataset": ds.full_dataset})


@fixture(scope="module")
def regr_dna_dataset_train(needed_dna_encodings: list[str]) -> Dataset:
    """Regression dataset - train split
    Args:
        needed_dna_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_dna_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset_train):
        ... "your function here"
        regr_dataset_train will use needed_dna_encodings
    """
    ds = RegressionDNAToyDataset(needed_dna_encodings)
    return ds.train


@fixture(scope="module")
def regr_dna_dataset_test(needed_dna_encodings: list[str]) -> Dataset:
    """Regression dataset - test split
    Args:
        needed_dna_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_dna_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset_test):
        ... "your function here"
        regr_dataset_test will use needed_dna_encodings
    """
    ds = RegressionDNAToyDataset(needed_dna_encodings)
    return ds.test


@fixture(scope="module")
def regr_dna_dataset(needed_dna_encodings: list[str]) -> Dataset:
    """Regression dataset - full dataset
    Args:
        needed_dna_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_dna_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset):
        ... "your function here"
        regr_dataset will use needed_dna_encodings
    """

    ds = RegressionDNAToyDataset(needed_dna_encodings)
    return ds.full_dataset


@fixture(scope="module")
def regr_dna_dataset_splits(needed_dna_encodings) -> Dataset:
    """Regression dataset - full dataset
    Args:
        needed_dna_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_dna_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset_splits):
        ... "your function here"
        regr_dataset_splits will use needed_dna_encodings
    """
    ds = RegressionDNAToyDataset(needed_dna_encodings)
    return ds.train_test_split_dataset


@fixture(scope="module")
def regr_dna_dataset_as_1split_dict(needed_dna_encodings) -> Dataset:
    """Regression dataset - full dataset
    Args:
        needed_dna_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split
    Note:
        To use this fixture, `needed_encoding` needs to be define in the unit test module as follows
        >>> @fixture(scope="module")
        >>> def needed_dna_encodings():
        ...    return ["aa_len", "aa_1hot"]
        It needs to return the list of needed encodings, this is just an example.
        >>> def test_function_in_specific_module(regr_dataset_splits):
        ... "your function here"
        regr_dataset_splits will use needed_dna_encodings
    """
    ds = RegressionDNAToyDataset(needed_dna_encodings)
    return DatasetDict({"whole_dataset": ds.full_dataset})

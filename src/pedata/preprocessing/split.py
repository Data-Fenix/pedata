"""
This file contains the abstract base class for splitters.
DatasetSplitter

it contains the following splitter classes:
- DatasetSplitterRandomTrainTest class for random train test split

"""

from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict, concatenate_datasets
from itertools import combinations
import numpy as np

def get_split_col_names(dataset: Dataset|DatasetDict) -> list[str]:
    """Return the split column names of a dataset
    Args:
        dataset: The dataset to process
    Returns:
        The split column names

    Example:
        >>> import datasets
        >>> from pedata.preprocessing.split import get_split_col_names
        >>> toy_dataset = datasets.Dataset.from_dict(
        ...    {
        ...         "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR"],
        ...         "target a": [1, 2, 3, 5, 4],
        ...         "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R"],
        ...         "split_random": ['train', 'train', 'train', 'test', 'test'],
        ...         "splittraintest": ['train', 'train', 'test', 'test', 'test'],
        ...         "RandomSplit": ['train', 'train', 'test', 'test', 'test'],
        ...         "source split": ['train', 'train', 'test', 'test', 'test'],
        ...         "split_k_fold": ['train', 'train', 'test', 'test', 'test'],
        ...         "split-k-fold": ['train', 'train', 'test', 'test', 'test'],
        ...     }
        ... )
        >>> get_split_col_names(toy_dataset)
        ['split_random', 'splittraintest', 'RandomSplit', 'source split', 'split_k_fold', 'split-k-fold']
    """
    if isinstance(dataset, DatasetDict):
        dataset = DatasetSplitter.concatenated_dataset(dataset)

    col_names = list(dataset.column_names)
    split_col_names = []
    for col_name in col_names:
        tokens_0 = col_name.split("_")
        tokens_1 = col_name.split("-")
        tokens_2 = col_name.split(" ")
        if 'split' in tokens_0 or 'split' in tokens_1 or 'split' in tokens_2:
            split_col_names.append(col_name)
        elif col_name.startswith('split'):
            split_col_names.append(col_name)
        elif 'Split' in col_name:
            split_col_names.append(col_name)
    
    return split_col_names

def get_split_map(splits: set[str], split_name_map: dict[str, str]) -> dict[str, str]:
    """Return the split map
    Args:
        splits: The splits
        split_name_map: The split name map
    Returns:
        The split map
    """
    sub_split_name_map = {}
    for split in splits:
        if split in split_name_map:
            sub_split_name_map[split] = split_name_map[split]
        elif 'train' in split and 'train'!=split:
            sub_split_name_map[split] = 'train'
        elif 'test' in split and 'test'!=split:
            sub_split_name_map[split] = 'test'
        elif 'val' in split and 'validation'!=split:
            sub_split_name_map[split] = 'validation'
    return sub_split_name_map

def rename_splits(dataset: Dataset|DatasetDict, split_name_map: dict[str, dict[str, str]]=None) -> Dataset|DatasetDict:
    """Rename the splits of a dataset
    Args:
        dataset: The dataset to process
        split_name_map: The split name map
    Returns:
        The modified dataset

    Example:
        >>> import datasets
        >>> from pedata.preprocessing.split import rename_splits
        >>> toy_dataset = datasets.Dataset.from_dict(
        ...    {
        ...         "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR"],
        ...         "target a": [1, 2, 3, 5, 4],
        ...         "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R"],
        ...         "split_random": ['training', 'training', 'training', 'testing', 'testing'],
        ...     }
        ... )
        >>> rename_splits(toy_dataset)['split_random']
        ['train', 'train', 'train', 'test', 'test']
    """
    if split_name_map is None:
        split_name_map = {}
        
    if isinstance(dataset, DatasetDict):
        for split_name in dataset.keys():
            dataset[split_name], split_name_map = rename_splits(dataset[split_name], split_name_map)
        splits = list(dataset)
        sub_split_name_map = get_split_map(splits, split_name_map.get("DatasetDict", {}))
        renamed_dataset_dict = {}
        for split in dataset:
            renamed_dataset_dict[sub_split_name_map.get(split, split)] = dataset[split]
        dataset = DatasetDict(renamed_dataset_dict)
        split_name_map["DatasetDict"] = sub_split_name_map
    else:
        split_col_names = get_split_col_names(dataset)
        for split_col_name in split_col_names:
            splits = set(dataset[split_col_name])
            sub_split_name_map = get_split_map(splits, split_name_map.get(split_col_name, {}))
            def mapper(sample):
                new_split_name = sub_split_name_map.get(sample[split_col_name], None)
                if new_split_name is not None:
                    sample[split_col_name] = new_split_name
                return sample
            dataset = dataset.map(mapper)
            split_name_map[split_col_name] = sub_split_name_map
    
    print("split name mapping during the preprocessing", split_name_map)
    return dataset, split_name_map

def split_col_histogram(dataset: Dataset|DatasetDict):
    """Return the split column histogram of a dataset
    
    Args:
        dataset: The dataset to process

    Returns:
        The split column histogram
    """
    histogram = {}
    if isinstance(dataset, DatasetDict):
        histogram = {"DatasetDict": {split: len(dataset[split]) for split in dataset}}
        dataset = DatasetSplitter.concatenated_dataset(dataset)

    split_col_names = get_split_col_names(dataset)
    for split_col_name in split_col_names:
        histogram[split_col_name] = {split: len(dataset.filter(lambda x: x[split_col_name]==split)) for split in set(dataset[split_col_name])}
    return histogram

class DatasetSplitter(
    ABC
):  # FIXME add a method to check and reorder the dataset in the exact same way it was before splitting
    """Dataset Splitter for random Train Test Split."""

    def __init__(
        self,
        append_split_col: bool = True,
    ) -> None:
        """Initialize the DatasetSplitter class.
        Args:
            dataset: The dataset to split
            seed: The seed to use for the split

        Raises:
            Type: If the dataset is not a Dataset or a DatasetDict
            Exception: If the dataset is a DatasetDict and contains more than one column
        Note:
            The Dataset can be a DatasetDict or a Dataset.
            If it is a DatasetDict, it should contain only one column, and it will be converted as Dataset.
        """
        self.append_split_col = append_split_col

    @property
    @abstractmethod
    def split_col_name(self) -> str:
        """Abstract method returning the name of the column containing the split name
        Returns:
            The name of the column containing the split name"""
        raise NotImplementedError
        return "awesome_split"

    @abstractmethod
    def _split(self, rng_seed: int = 0) -> DatasetDict:
        """Implementing the dataset split method

        Args:
            rng_seed: The seed to use for the random generator used in the method (if any)
        Returns:
            The split dataset"""

        raise NotImplementedError

    def split(
        self,
        dataset: DatasetDict | Dataset,
        rng_seed: int = 0,
        return_dataset_dict=True,
    ) -> DatasetDict | Dataset:
        """Split the dataset and return it.
        Args:
            dataset: The dataset to split -
                if it is a DatasetDict, it should contain only one split
            return_dataset_dict: Whether to return a concatenated dataset or a DatasetDict
                If True -> return concatenated dataset -> Dataset
                If False, return a DatasetDict with the split name as key (Default)
        Return:
            The split dataset as a DatasetDict
        """
        # defines the rng

        # sets the return_dataset_dict attribute
        self._return_dataset_dict = return_dataset_dict

        # Checking if the dataset is a Dataset or a DatasetDict of only one split
        self._dataset_check(dataset)

        # Makes sure that ._dataset is of type Dataset"""
        self._dataset_preprocessing(dataset)

        if self.split_col_name not in self._dataset.column_names:
            # if the split column is not in the dataset, create the split
            self._split_dataset = self._split(rng_seed)  # split the dataset

        else:
            self.append_split_col = False
            # if the split column is in the dataset, use it to split the dataset
            self._split_dataset = self.split_dataset_using_split_column(
                self._dataset, self.split_col_name
            )

        if self.append_split_col:
            # append the split column to the _split_dataset
            self._split_dataset = self.append_split_column(
                self._split_dataset, self.split_col_name
            )
            # update _dataset with the split column
            self._dataset = self.concatenated_dataset(self._split_dataset)

        return self.dataset

    @staticmethod
    def _dataset_check(dataset: DatasetDict | Dataset):
        """Checking if the dataset is a Dataset or a DatasetDict of only one split
        Args:
            dataset: The dataset to check
        Raises:
            TypeError: If the dataset is not a Dataset or a DatasetDict
            ValueError: If the dataset is a DatasetDict and contains more than one column
        """
        if not (isinstance(dataset, DatasetDict) or isinstance(dataset, Dataset)):
            raise TypeError(
                f"dataset must be a DatasetDict or a Dataset, got {type(dataset)} instead."
            )

        # if the dataset is a DatasetDict, convert it to a Dataset
        if isinstance(dataset, DatasetDict):
            # The input DatasetDict Should have only one split
            if len(dataset) > 1:
                raise ValueError(
                    f"A DatasetDict can be provided as an input, however it should contain only one split. "
                    f"Here the dataset contains {len(dataset)} splits: "
                    f"{[split for split in dataset]}"
                )

    def _dataset_preprocessing(self, dataset: Dataset | DatasetDict) -> None:
        """Makes sure that ._dataset is a Dataset"""
        if isinstance(dataset, DatasetDict):
            # concatenate the dataset as a Dataset
            dataset = self.concatenated_dataset(dataset)

        self._dataset = dataset

    @property
    def dataset(self) -> Dataset | DatasetDict:
        """Return the dataset.
        Returns:
            The dataset as a DatasetDict or Dataset
        Note:
            If self.return_dataset_dict is True, return a DatasetDict
            If self.return_dataset_dict is False, return a Dataset
        """
        if self._return_dataset_dict:
            return self._split_dataset
        else:
            return self._dataset

    @staticmethod
    def split_dataset_using_split_column(
        dataset: Dataset, split_col_name: str
    ) -> DatasetDict:
        """Split the dataset using a column containing the split name.
        Args:
            dataset: The dataset to split
            split_col_name: The name of the column containing the split name
        Returns:
            DatasetDict containing the split dataset
        Raises:
            TypeError: If the dataset is not a Dataset
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be a Dataset, got {type(dataset)} instead.")

        split_dataset = DatasetDict()
        splits = sorted(list(set(dataset[split_col_name])))
        for split in splits:
            split_dataset[split] = dataset.filter(lambda x: x[split_col_name] == split)
        return split_dataset

    @staticmethod
    def concatenated_dataset(
        dataset: DatasetDict, split_list: list[str] = []
    ) -> Dataset:
        """Return the concatenated dataset.
        Args:
            dataset: The dataset to concatenate.
            split_list: The list of splits to concatenate
        Returns:
            The concatenated dataset
        """
        if not isinstance(dataset, DatasetDict):
            raise TypeError(
                f"dataset must be a DatasetDict, got {type(dataset)} instead."
            )

        if len(split_list) == 0:
            split_list = list(dataset.keys())

        return concatenate_datasets(
            [dataset[split] for split in dataset if split in split_list]
        )

    @staticmethod
    def append_split_column(dataset: DatasetDict, split_col_name: str) -> DatasetDict:
        """Add a column to the dataset with the name of the split.
        Args:
            dataset: The dataset to modify
            split_col_name: The name of the column containing the split name
        Returns:
            The modified dataset
        Raises:
            Exception: If the split column was not added correctly - this should never happen but as quite critical, we check it anyway
        """
        for split_name, dataset_split in dataset.items():
            # if the split column is already in the dataset, no need to add it again
            if split_col_name in dataset_split.column_names:
                continue
            split_col = [split_name for _ in range(dataset_split.num_rows)]
            dataset[split_name] = dataset_split.add_column(split_col_name, split_col)

        for split_name, dataset_split in dataset.items():
            assert (
                split_col_name in dataset_split.column_names
            ), f"The split column {split_col_name} was not added correctly to the dataset {split_name}"

        return dataset

    @staticmethod
    def yield_all_train_test_sets(
        dataset: DatasetDict, combined_n: int = 1
    ) -> DatasetDict:
        """Yield all train test splits from a k-fold dataset
        Args:
            dataset: The dataset dictionnary to process
                It must contain items with keys such as "split_1", "split_2", ..., "split_k"
            combine_n: The number of splits to combine for the TEST set
        Yields:
            The train test splits
        Note:
            If the dataset is already a train test split, the train and test sets will simply be yielded

        Example:
            >>> import datasets
            >>> from pedata.preprocessing.split import DatasetSplitterRandomKFold
            >>> toy_dataset = datasets.Dataset.from_dict(
            ...    {
            ...         "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR", "RLGLYITR"],
            ...         "target a": [1, 2, 3, 5, 4, 6],
            ...         "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R", "A2L_M1R"],
            ...     }
            ... )
            >>> splitter = DatasetSplitterRandomKFold(toy_dataset, k=3)
            >>> dataset = splitter.split()
            >>> k = 0
            >>> for train_test_set in splitter.yield_all_train_test_sets(dataset, combined_n=1):
            ...     k += 1
            ...     print("________________")
            ...     print(f"- train_test_{k} -")
            ...     print(train_test_set)
            ________________
            - train_test_1 -
            DatasetDict({
                train: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 4
                })
                test: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 2
                })
            })
            ________________
            - train_test_2 -
            DatasetDict({
                train: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 4
                })
                test: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 2
                })
            })
            ________________
            - train_test_3 -
            DatasetDict({
                train: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 4
                })
                test: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 2
                })
            })
        """

        # Generate all combinations of combined_n elements from the list of splits
        dataset_splits = sorted(list(dataset.keys()))
        if dataset_splits != ["test", "train"]:
            combinations_list = list(combinations(dataset_splits, combined_n))
            # Yield all combinations as train test splits
            for combination in combinations_list:
                test = concatenate_datasets([dataset[j] for j in combination])
                train = concatenate_datasets(
                    [dataset[j] for j in dataset_splits if j not in combination]
                )
                yield DatasetDict({"train": train, "test": test})
        else:  # if the dataset is already a train test split, yield it
            yield dataset


class DatasetSplitterRandomTrainTest(DatasetSplitter):
    """Dataset Splitter for random Train Test Split."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @property
    def split_col_name(self) -> str:
        """Abstract method returning the name of the column containing the split name
        Retunrs:
            The name of the column containing the split name"""
        return "random_split_train_0_8_test_0_2"

    def _split(self, rng_seed: int = 0) -> DatasetDict:
        """Split the dataset into train and test sets.
        Returns:
            The split dataset as a DatasetDict
        Example:#FIXME: fix the example rng
        >>> import datasets
        >>> toy_dataset = datasets.Dataset.from_dict(
        ...    {
        ...         "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR"],
        ...         "target a": [1, 2, 3, 5, 4],
        ...         "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R"],
        ...     }
        ... )
        >>> splitter = DatasetSplitterRandomTrainTest(toy_dataset)
        >>> dataset = splitter.split()
        >>> print(dataset)
        DatasetDict({
            train: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_train_0_8_test_0_2'],
                num_rows: 4
            })
            test: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_train_0_8_test_0_2'],
                num_rows: 1
            })
        })
        """
        return self._dataset.train_test_split(
            test_size=0.2, generator=np.random.default_rng(rng_seed)
        )


class DatasetSplitterRandomKFold(DatasetSplitter):
    """Dataset Splitter for random Train Test Split."""

    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.k = k

    @property
    def split_col_name(self) -> str:
        """Abstract method returning the name of the column containing the split name
        Retunrs:
            The name of the column containing the split name"""
        return f"random_split_{self.k}_fold"

    def _split(self, rng_seed: int = 0) -> DatasetDict:
        """Split the dataset into K folds. #FIXME: fix the example rng
        Returns:
            The split dataset as a DatasetDict

        Raises:
            ValueError: If k is larger than the number of datapoints in the dataset

        Example:
        >>> import datasets
        >>> from pedata.preprocessing import DatasetSplitterRandomKFold
        >>> toy_dataset = datasets.Dataset.from_dict(
        ...    {
        ...         "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR", "RLGLYITR"],
        ...         "target a": [1, 2, 3, 5, 4, 6],
        ...         "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R", "A2L_M1R"],
        ...     }
        ... )
        >>> splitter = DatasetSplitterRandomKFold(toy_dataset, k=3)
        >>> dataset = splitter.split()
        >>> print(dataset)
        DatasetDict({
            split_0: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                num_rows: 2
            })
            split_1: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                num_rows: 2
            })
            split_2: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                num_rows: 2
            })
        })
        """
        # check if k is compatible with the dataset
        if self.k > len(self._dataset):
            raise ValueError(
                f"k must be smaller than or equal to the number of datapoints in the dataset, "
                f"Here, we got k = {self.k} and {len(self._dataset)} datapoints respectively."
            )

        # shuffle the dataset
        self._dataset = self._dataset.shuffle(generator=np.random.default_rng(rng_seed))

        # calculate fold size
        fold_size = len(self._dataset) // self.k  # determine the fold size

        # Create k folds
        split_dataset = {}
        for i in range(self.k):
            # Determine the start and end index for the validation set
            start_index = i * fold_size
            end_index = (i + 1) * fold_size if i < self.k - 1 else len(self._dataset)
            split_dataset[f"split_{i}"] = Dataset.from_dict(
                self._dataset[start_index:end_index]
            )

        return DatasetDict(split_dataset)


class ZeroShotSpliter(DatasetSplitter):
    """
    Dataset Splitter for zero shot datasets.
    Zero Shot datasets have manually defined splits instead of random splits.
    This splitter splits the datasets using a defined split column or using the existing
    hugging face splits
    """

    def __init__(self, split_col_name: str):
        super().__init__()
        self._split_col_name = split_col_name

    @property
    def split_col_name(self) -> str:
        return self._split_col_name

    def _split(self, rng_seed: int = 0) -> DatasetDict:
        """
        Notes:
            This method should not be called for zero shot datasets.
        """
        raise ValueError(
            "Zero shot datasets should not be split using this splitter. It should contain the split column"
        )


def append_split_columns_to_dataset(
    dataset: Dataset,
) -> Dataset:
    """Append all split columns to a dataset
    Args:
        dataset: The dataset to modify
    Returns:
        The modified dataset
    """
    # Split the dataset into train and test sets
    dataset = DatasetSplitterRandomTrainTest().split(
        dataset, rng_seed=12053, return_dataset_dict=False
    )
    # Split the dataset into k folds
    k = (
        10 if len(dataset) >= 10 else len(dataset)
    )  # hack for very small dataset (testing purposes)

    dataset = DatasetSplitterRandomKFold(k=k).split(
        dataset, rng_seed=12053, return_dataset_dict=False
    )

    return dataset


def add_source_split_column(
    dataset: DatasetDict, merge_split: bool = False
) -> DatasetDict:
    """Add a column to the dataset with the name of the splits used by the source of the dataset.
    Args:
        dataset: The dataset to modify
        merge_split: Whether to merge the split columns into one column
    Returns:
        The modified dataset
    """
    # instantiate a splitter - won't be used to split but
    dataset = DatasetSplitter.append_split_column(dataset, "source_split")

    # merge the splits into one column if merge_split is True
    if merge_split:
        dataset = DatasetDict(
            {"whole_dataset": DatasetSplitter.concatenated_dataset(dataset)}
        )

    return dataset

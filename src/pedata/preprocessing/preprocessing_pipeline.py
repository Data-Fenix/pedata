# data
from datasets import Dataset, DatasetDict

# pedata
from . import append_split_columns_to_dataset
from . import append_index_column_to_dataset
from ..integrity import check_dataset_format, check_char_integrity_in_sequences
from ..mutation import fill_all_missing_sequences

# from . import tag_finder


def preprocessing_pipeline(
    dataset: Dataset,
    invalid_data_removal: bool = True,
    fill_missing_sequences: bool = True,
    mutation_code_offset: int = None,
    add_index: bool = False,
    tag_finder: bool = False,
    add_splits: bool = False,
) -> Dataset:
    """Perform preprocessing on the dataset
    Args
        dataset: dataset to process
        invalid_data_removal: whether to remove invalid data from the dataset
            if `False` will return an error if a character is not in the alphabet
            otherwise will remove the invalid datapoints
        fill_missing_sequences: whether to fill missing sequences.
            The mutation code is used in combination with the parent sequence and the offset to fill the missing sequences
            The parent sequence should be in the dataset; the mutation code offset can be detemrined automatically or provided (mutation_code_offset)
        mutation_code_offset: offset to apply to the missing values
            Will be automatically determined if not provided
        add_index: whether to add an index column to the dataset
        tag_finder: whether to process dataset using tag_finder
        add_splits: whether to add split columns to the dataset
        mutation_code_offset: offset to apply to the missing values
            Default to None, meaning that the mutation_code_offset will be determined automatically through the mutation code.
    Returns:
        preprocessed dataset
    Raises:
        TypeError: If the input is not a valid dataset or dictionary of datasets.
    """
    if not (
        isinstance(dataset, Dataset)
        or (
            isinstance(dataset, DatasetDict)
            and len(list(dataset.column_names.keys())) == 1
        )
    ):  # FIXME: test this - input a datasetdict with more than one split - should raise an error
        raise TypeError(
            f"Input a valid dataset -> datasets.Dataset or datasets.DatasetDict with only one split - here is {type(dataset)}"
        )

    if isinstance(dataset, DatasetDict):
        dataset = dataset[list(dataset.column_names.keys())[0]]

    # check dataset integrity
    dataset = check_dataset_format(dataset)

    # check sequence integrity
    dataset, rows_without_errors = check_char_integrity_in_sequences(
        dataset=dataset,
        invalid_data_padding=invalid_data_removal,
    )

    # get rid of the errors
    if invalid_data_removal:
        dataset = dataset.select(rows_without_errors)

    # fill missing sequences
    if fill_missing_sequences:
        dataset = fill_all_missing_sequences(
            dataset, mutation_code_offset=mutation_code_offset
        )

    if add_index:
        # Add index column to dataset
        dataset = append_index_column_to_dataset(dataset)

    if tag_finder:
        # process dataset using tag_finder #TODO - first test tag_finder
        if False:
            dataset = tag_finder(dataset)

    if add_splits:
        # Add split columns to dataset
        dataset = append_split_columns_to_dataset(
            dataset,
        )

    return dataset

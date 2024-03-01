# system
import os

# dataset
from datasets import Dataset
import pandas as pd

# peadata
from ..hfhub_tools.utils import check_if_dataset_on_hf, get_dataset


def load_data(dataname: str) -> Dataset:
    """load dataset from file of from HuggingFace

    Args:
        filename: name of the file / dataset to process
            It can be a path to a file, a name of a dataset in the standard datasets directory or a name of a dataset on HuggingFace
    Returns:
        the dataset
    """

    if os.path.exists(dataname):
        dataset = _read_dataset_from_file(dataname)
    elif check_if_dataset_on_hf(dataname):
        dataset = get_dataset(dataname)
    else:
        print(check_if_dataset_on_hf(dataname))
        raise ValueError(f"Data file '{dataname}' not found")

    return dataset


def _read_dataset_from_file(filename: str) -> Dataset:
    """Reads a CSV, Excel or Parquet file and return a HuggingFace dataset.
    Args:
        filename: File name of the source CSV file.

    Returns:
        A HuggingFace dataset.

    Raises:
        TypeError: If the input type is not a CSV, Excel or Parquet file.

    """
    filename = str(filename).lower()

    # Check file format
    if filename.endswith("csv"):
        df = _read_csv_ignore_case(filename)

    elif filename.endswith("xls") or filename.endswith("xlsx"):
        df = pd.read_excel(filename, 0)

    elif filename.endswith("parquet"):
        df = pd.read_parquet(filename)

    else:
        raise TypeError("Invalid input: input either a csv, excel or parquet file")

    return Dataset.from_pandas(df)


def _read_csv_ignore_case(file_path: str) -> pd.DataFrame:
    """Reads a CSV file with a case-insensitive match

    Args:
        file_path: path to the file

    Returns:
        The dataframe

    Raises:
        FileNotFoundError if no file is matching
    """
    directory, file_name = os.path.split(file_path)
    if len(directory) == 0:
        directory = os.getcwd()
    # List all files in the directory
    files_in_directory = os.listdir(directory)

    # Find the file with a case-insensitive match
    matching_files = [
        file for file in files_in_directory if file.lower() == file_name.lower()
    ]

    if not matching_files:
        raise FileNotFoundError(f"No file found matching: {file_name}")

    # Use the first matching file (in case there are multiple matches)
    matching_file_path = os.path.join(directory, matching_files[0])

    # Use pd.read_csv with the found file path
    return pd.read_csv(matching_file_path)

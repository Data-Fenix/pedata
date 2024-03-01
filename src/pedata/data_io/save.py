import os
from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd


def save_dataset_as_csv(
    dataset: DatasetDict | Dataset | pd.DataFrame,
    filename: str | Path,
) -> None:
    """Saves the dataset as csv

    Args:
        dataset: dataset to save
        filename: filename to save the dataset
    """
    if isinstance(dataset, DatasetDict):
        for split in dataset.keys():
            save_dataset_as_csv(
                dataset[split], filename=f"{filename.split('.')[0]}_{split}.csv"
            )
    else:
        if isinstance(filename, str):
            filename = Path(os.path.abspath(filename))

        if isinstance(dataset, Dataset):
            # convert Dataset to pandas dataframe
            dataset = dataset.to_pandas()

        # save as csv
        dataset.to_csv(filename, index=False)

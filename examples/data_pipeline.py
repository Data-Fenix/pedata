from pedata.data_io import load_data
from pedata.integrity import check_dataset_format
from pedata.preprocessing import preprocessing_pipeline
from pedata.encoding import add_encodings


def pipeline(datasource, needed=["aa_len"]) -> None:
    """Pipeline from local datafile"""
    dataset = load_data(datasource)
    dataset = check_dataset_format(dataset)
    dataset = preprocessing_pipeline(dataset)
    dataset = add_encodings(dataset, needed=needed)
    print(f"- {datasource=} - {needed=}")
    print(dataset.to_pandas())


def main():
    pipeline("examples/example_dataset.csv")
    pipeline("Exazyme/CrHydA1_PE_REGR")


if __name__ == "__main__":
    main()

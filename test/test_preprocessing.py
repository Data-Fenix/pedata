from pedata.preprocessing import preprocessing_pipeline, append_index_column_to_dataset
from pytest import fixture
import pytest

from pedata.static import (
    dna_example_1_missing_val,
    dna_example_1_no_missing_val,
    aa_example_0_missing_val,
    aa_example_0_no_missing_val,
    aa_example_0_invalid_alphabet,
)


@fixture(scope="module")
def needed_encodings():
    return ["aa_seq", "aa_1hot"]


@fixture(scope="module")
def aa_expected_features():
    return [
        "aa_mut",
        "target_kcat_per_kmol",
        "aa_seq",
        "aa_1hot",
    ]


def test_preprocessing_pipeline(regr_dataset, aa_expected_features):
    """test that all default features are added as expected by the preprocessing_pipeline"""

    # default preprocessing and transform
    dataset = preprocessing_pipeline(regr_dataset)
    assert all([feature in dataset.features for feature in aa_expected_features])

    dataset = preprocessing_pipeline(regr_dataset, add_index=False, add_splits=True)
    assert "index" not in dataset.features
    assert "random_split_train_0_8_test_0_2" in dataset.features
    assert "random_split_10_fold" in dataset.features

    dataset = preprocessing_pipeline(regr_dataset, add_index=True, add_splits=True)
    assert "index" in dataset.features
    assert "random_split_train_0_8_test_0_2" in dataset.features
    assert "random_split_10_fold" in dataset.features


def test_preprocessing_pipeline_raises_error(regr_dataset_splits):
    """testing that errors are raised"""
    # if datasetDict with more than one split as input
    with pytest.raises(TypeError):
        preprocessing_pipeline(regr_dataset_splits)


def test_append_index_column_to_dataset(regr_dataset):
    """Test that the append_index_column_to_dataset function works"""

    # ammends the dataset with an index column
    dataset_ammended = append_index_column_to_dataset(regr_dataset)
    assert "index" in dataset_ammended.column_names

    # does not ammend the dataset if it already has an index column"""
    dataset_ammended_2 = append_index_column_to_dataset(dataset_ammended)
    assert dataset_ammended == dataset_ammended_2


def test_append_index_column_to_dataset_with_incorrect_dataset(
    regr_dataset_as_1split_dict,
):
    """Test that the append_index_column_to_dataset returns TypeError if the input is incorrect"""
    with pytest.raises(TypeError):
        _ = append_index_column_to_dataset(regr_dataset_as_1split_dict)


# fixture for preprocessing_pipeline
@fixture(scope="module")
def encodings_to_add():
    return ["aa_1gram", "aa_unirep_1900"]


# def test_preprocessing_pipeline_with_incorrect_dataset(
#     regr_dataset_splits, encodings_to_add
# ):
#     """Test that the test_preprocessing_pipeline returns TypeError if the input is incorrect"""
#     with pytest.raises(TypeError):
#         _ = preprocessing_pipeline(regr_dataset_splits, needed_encodings=encodings_to_add)


# def test_preprocessing_pipeline_with_correct_datasetDict(
#     regr_dataset_as_1split_dict, encodings_to_add
# ):
#     """Test that the test_preprocessing_pipeline returns TypeError if the input is incorrect"""
#     _ = preprocessing_pipeline(
#         regr_dataset_as_1split_dict, needed_encodings=encodings_to_add
#     )


if False:

    def test_preprocess_data_aa(aa_expected_new_features):
        """preprocess_data  - aa"""
        # Missing values in "aa_seq" column
        pd.DataFrame(aa_example_0_missing_val).to_csv("tempfile.csv", index=False)
        dataset = preprocess_data("tempfile.csv")
        clean_up()
        assert all(
            feature in list(dataset.features.keys())
            for feature in aa_expected_new_features
        )

        # No missing values in "aa_seq" column
        pd.DataFrame(aa_example_0_no_missing_val).to_csv("tempfile.csv", index=False)
        dataset = preprocess_data("tempfile.csv")
        clean_up()
        assert all(
            feature in list(dataset.features.keys())
            for feature in aa_expected_new_features
        )

    def test_preprocess_data_dna(dna_expected_new_features):
        """preprocess_data dna"""
        # Missing values in "dna_seq" column
        pd.DataFrame(dna_example_1_missing_val).to_csv("tempfile.csv", index=False)
        dataset = preprocess_data("tempfile.csv")
        clean_up()
        assert all(
            [
                feature in list(dataset.features.keys())
                for feature in dna_expected_new_features
            ]
        )

        # No missing values in "dna_seq" column
        pd.DataFrame(dna_example_1_no_missing_val).to_csv("tempfile.csv", index=False)
        dataset = preprocess_data("tempfile.csv")
        clean_up()
        assert all(
            [
                feature in list(dataset.features.keys())
                for feature in dna_expected_new_features
            ]
        )

    def test_preprocess_data_invalid_alphabet():
        """preprocess_data test: Containing invalid AA alphabets"""
        pd.DataFrame(aa_example_0_invalid_alphabet).to_csv("tempfile.csv", index=False)
        with pytest.raises(ValueError):
            _ = preprocess_data("tempfile.csv")

        clean_up()

    def test_preprocess_data_local_filesytem(aa_expected_new_features, folder_path):
        """preprocess_data test: using local file system"""
        pd.DataFrame(aa_example_0_no_missing_val).to_csv("tempfile.csv", index=False)
        local_filesystem = fsspec.filesystem("file")
        dataset = preprocess_data(
            "tempfile.csv", save_to_path=folder_path, filesystem=local_filesystem
        )
        clean_up()
        assert all(
            feature in list(dataset.features.keys())
            for feature in aa_expected_new_features
        )

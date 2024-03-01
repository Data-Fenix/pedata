# datasets
import datasets as ds

# pedata
from pedata.integrity import check_dataset_format, check_char_integrity_in_sequences

# pytest
import pytest


def test_check_dataset():
    # Test case 1: Invalid mutation
    invalid_mut = {"aa_mut": ["MLDSWE"], "aa_seq": [None]}  # Dictionary
    with pytest.raises(TypeError):
        check_dataset_format(invalid_mut)

    # Test case 3: Missing "aa_seq" column
    invalid_dataset = ds.Dataset.from_dict({"aa_mut": [], "target": []})
    with pytest.raises(KeyError):
        check_dataset_format(invalid_dataset)

    # Test case 4: Missing "dna_seq" column
    invalid_dataset = ds.Dataset.from_dict({"dna_mut": [], "target": []})
    with pytest.raises(KeyError):
        check_dataset_format(invalid_dataset)

    # Test case 5: Missing "aa_mut" column
    invalid_dataset = ds.Dataset.from_dict(
        {"aa_seq": [None, None], "target foo": [1, 2]}
    )
    with pytest.raises(KeyError):
        check_dataset_format(invalid_dataset)

    # Test case 6: Missing "dna_mut" column
    invalid_dataset = ds.Dataset.from_dict(
        {"dna_seq": [None, None], "target foo": [1, 2]}
    )
    with pytest.raises(KeyError):
        check_dataset_format(invalid_dataset)

    # Test case 7: Missing a column starting with keyword "target"
    invalid_dataset = ds.Dataset.from_dict(
        {"dna_mut": ["A3T", "T2A"], "dna_seq": [None, None]}
    )
    with pytest.raises(KeyError):
        check_dataset_format(invalid_dataset)

    # Test case 8: "target summary variable" column is present
    invalid_dataset = ds.Dataset.from_dict(
        {"dna_seq": ["ATGCTACG", "ATGCAACG"], "target summary variable": [1, 2]}
    )
    with pytest.raises(KeyError):
        check_dataset_format(invalid_dataset)

    # Test case 9: "aa_seq" has no missing values
    valid_dataset = ds.Dataset.from_dict(
        {"aa_seq": ["MLGLYITR", "MAGLYITR"], "target foo": [1, 2]}
    )

    # Test case 10: "dna_seq" has no missing values
    valid_dataset = ds.Dataset.from_dict(
        {"dna_seq": ["ATGCTACG", "ATGCAACG"], "target foo": [1, 2]}
    )
    check_dataset_format(valid_dataset)

    # Test case 11: "aa_mut" column has missing values
    valid_dataset = ds.Dataset.from_dict(
        {
            "aa_seq": ["MLGLYITR", None],
            "aa_mut": ["wildtype", "L2R"],
            "target foo": [1, 2],
        }
    )
    check_dataset_format(valid_dataset)

    # Test case 12: "dna_mut" column has missing values
    valid_dataset = ds.Dataset.from_dict(
        {
            "dna_seq": ["ATGCTACG", None],
            "dna_mut": ["wildtype", "T2A"],
            "target foo": [1, 2],
        }
    )
    check_dataset_format(valid_dataset)

    # Test case 13: Missing column starting with keyword "target"
    invalid_dataset = ds.Dataset.from_dict(
        {
            "dna_seq": ["ATGCTACG", None],
            "dna_mut": ["wildtype", "T2A"],
        }
    )
    with pytest.raises(KeyError):
        check_dataset_format(invalid_dataset)


def test_check_char_integrity_in_sequences():
    # Test case 14: Wrong Vocabulary
    invalid_dataset = ds.Dataset.from_dict(
        {
            "dna_seq": ["ATGCTXCG", None],
            "dna_mut": ["wildtype", "T2A"],
            "target foo": [1, 2],
        }
    )
    with pytest.raises(ValueError):
        check_char_integrity_in_sequences(invalid_dataset, invalid_data_padding=False)

    # Test case 15: Wrong Vocabulary
    invalid_dataset = ds.Dataset.from_dict(
        {
            "aa_seq": ["MLGLYIXR", "MRGLYIRR"],
            "aa_mut": ["wildtype", "L2R"],
            "target foo": [1, 2],
        }
    )
    with pytest.raises(ValueError):
        check_char_integrity_in_sequences(invalid_dataset, invalid_data_padding=False)

    # replace the error by padding_value_enc
    dataset, rows_without_errors = check_char_integrity_in_sequences(
        invalid_dataset, invalid_data_padding=True
    )
    assert len(rows_without_errors) == 1
    # this time it should not find any errors
    dataset, rows_without_errors = check_char_integrity_in_sequences(
        dataset, invalid_data_padding=True
    )
    assert len(rows_without_errors) == 2

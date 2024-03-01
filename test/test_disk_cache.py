import pytest
from pedata.disk_cache import load_similarity

from pytest import fixture


# ======== Fixtures ======
@fixture(scope="module")
def aa_expected_new_features():
    return [
        "aa_unirep_1900",
        "aa_unirep_final",
        "aa_len",
        "aa_1gram",
        "aa_1hot",
    ]


@fixture(scope="module")
def dna_expected_new_features():
    return ["dna_len", "dna_1hot"]


@fixture(scope="module")
def alphabet_type_dna():
    return "dna"


@fixture(scope="module")
def alphabet_type_aa():
    return "aa"


@fixture(scope="module")
def folder_path():
    return "processed_data"


@fixture(scope="module")
def needed_encodings():
    return ["aa_seq", "aa_1hot"]


# ======== Tests ======


def test_load_similarity_dna(alphabet_type_dna):
    """Test load_similarity with DNA alphabet"""
    # Load single DNA similarity matrix and replace disk cache
    similarity_name = "Simple"
    _, similarity_matrix = load_similarity(
        alphabet_type_dna, similarity_name, replace_existing=True
    )
    assert similarity_matrix[0][0] == 1 and similarity_matrix[-1][-1] == 1

    # Load multiple DNA similarity matrix and replace disk cache
    similarity_name = ["Simple", "Identity"]
    _, similarity_matrix = load_similarity(
        alphabet_type_dna, similarity_name, replace_existing=True
    )
    assert similarity_matrix[0][0][-1] == 0 and similarity_matrix[1][2][2] == 1

    #  Without overwriting an existing DNA similarity matrix (replace_existing=False)
    similarity_name = "Simple"
    _, similarity_matrix = load_similarity(
        alphabet_type_dna, similarity_name, replace_existing=False
    )
    assert similarity_matrix[0][0] == 1 and similarity_matrix[-1][-1] == 1


def test_load_similarity_aa(alphabet_type_aa):
    """Test load_similarity with AA alphabet"""
    # Load single AA similarity matrix and replace disk cache
    similarity_name = "BLOSUM62"
    _, similarity_matrix = load_similarity(
        alphabet_type_aa, similarity_name, replace_existing=True
    )
    assert similarity_matrix[0][0] == 4 and similarity_matrix[13][4] == -3

    # Load multiple AA similarity matrix and replace disk cache
    similarity_name = ["BLOSUM62", "BLOSUM90", "IDENTITY", "PAM20"]
    _, similarity_matrix = load_similarity(
        alphabet_type_aa, similarity_name, replace_existing=True
    )
    assert (
        similarity_matrix[0][0][0] == 4
        and similarity_matrix[1][10][10] == 7
        and similarity_matrix[2].shape[0] == 21
        and similarity_matrix[3][0][4] == -9
    )

    # Without overwriting an existing AA similarity matrix (replace_existing=False)
    similarity_name = "PAM20"
    _, similarity_matrix = load_similarity(
        alphabet_type_aa, similarity_name, replace_existing=False
    )
    assert similarity_matrix[0][4] == -9 and similarity_matrix[-1][1] == -19


def test_load_similarity_invalid_alphabet():
    """Test load_similarity: Invalid alphabet type"""
    alphabet_type_bb = "bb"
    similarity_name = "Simple"
    with pytest.raises(ValueError):
        load_similarity(alphabet_type_bb, similarity_name)


if False:
    # ======== Tests to implement ======
    def test_load_similarity_dim_and_constency():
        """Raised error when the dimensions and consistency of the matrix is not as expected"""
        raise (NotImplementedError)

    def test_load_similarity_missing_entries():
        """Raised error for missing entries in the similarity matrix"""
        raise (NotImplementedError)

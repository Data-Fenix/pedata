__all__ = [
    "check_dataset_format",
    "check_mutation_namedtuple",
    "check_char_integrity_in_sequences",
]

from datasets import Dataset
from .constants import Mut, Mut_with_no_targ
from .encoding.alphabets import (
    padding_value_enc,
    valid_aa_alphabet,
    valid_dna_alphabet,
    valid_smiles_alphabet,
)


def check_dataset_format(dataset: Dataset) -> Dataset:
    """
    Function checks if the dataset is valid.

    Args:
        dataset: Dataset to check. It should be an instance of Dataset.

    Raises:
        TypeError: If the input dataset is not an instance of Dataset.
        KeyError: If the dataset is missing a required column or contains a non-allowed column.

    Example:
        >>> dataset = Dataset.from_dict({"aa_seq": [None, None]})
        KeyError: Columns are missing in the dataset. In particular: aa_mut.

    """

    if isinstance(dataset, Dataset):
        feature_keys = dataset.features.keys()
    else:
        raise TypeError("Mutation should be a dataset")
    dataset = dataset.with_format("pandas")
    missing_cols = []

    # Check if either "aa_seq" or "dna_seq" column is missing
    if all(
        [
            "aa_seq" not in feature_keys,
            "dna_seq" not in feature_keys,
            "smiles_seq" not in feature_keys,
        ]
    ):
        missing_cols.append('"aa_seq" or "dna_seq" or "smiles_seq"')

    else:
        # Calculate the number of missing values in either "aa_seq" or "dna_seq" column
        if "aa_seq" in feature_keys:
            missing_values = dataset["aa_seq"].isnull().sum()

            # Check if missing values exist and "aa_mut" column is missing
            if missing_values > 0 and "aa_mut" not in feature_keys:
                missing_cols.append("aa_mut")

        if "dna_seq" in feature_keys:
            missing_values = dataset["dna_seq"].isnull().sum()

            # Check if missing values exist and "aa_mut" column is missing
            if missing_values > 0 and "dna_mut" not in feature_keys:
                missing_cols.append("dna_mut")

        if "smiles_seq" in feature_keys:
            pass  # nothing to do here

    # Check if "target summary variable" column is present
    if "target summary variable" in feature_keys:
        raise KeyError(
            "There was already a column called 'target summary variable' in the data set. This is a special column name reserved for Exazymes internal use."
        )

    # Check if any column starts with "target"
    if len([k for k in feature_keys if k.lower().startswith("target")]) == 0:
        missing_cols.append('a column starting with "target"')

    # Raise KeyError if any missing columns are found
    if len(missing_cols) > 0:
        raise KeyError(
            f"Columns are missing in the data file. In particular: {', '.join(missing_cols)}."
        )

    return dataset


# Validate namedtuple mutation
def check_mutation_namedtuple(m: Mut):
    """
    Function validates a namedtuple mutation.

    Args:
        m (Mut): The namedtuple mutation to validate.

    Raises:
        TypeError: If the mutation is not a valid namedtuple or if attributes are of incorrect types.

    Example:
        >>> mutation = Mut(pos='2', src='A', targ='C')
        >>> check_mutation_namedtuple(mutation)
        TypeError: Attribute 'pos' should exist in a namedtuple and be an int
    """

    if not (isinstance(m, Mut) or isinstance(m, Mut_with_no_targ)) or len(m) < 2:
        raise TypeError(
            "Invalid format. Each mutation namedtuple should be an instance of a Mut namedtuple with at least two attributes: (pos, src), and atmost 3 attributes: (pos, src, targ)"
        )

    if not hasattr(m, "pos") or not isinstance(m.pos, int):
        raise TypeError("Attribute 'pos' should exist in a namedtuple and be an int")

    if not hasattr(m, "src") or not isinstance(m.src, str):
        raise TypeError("Attribute 'src' should exist in a namedtuple and be a string")

    if len(m) == 3:
        if not hasattr(m, "targ") or not isinstance(m.targ, str):
            raise TypeError(
                "If a namedtuple has length of 3, attribute 'targ' should exist as a string"
            )
    if hasattr(m, "src") and not len(m.src) == 1:
        raise TypeError(f"Mutation source '{m.src}' is not a single character.")

    if hasattr(m, "targ") and not len(m.targ) == 1:
        raise TypeError(f"Mutation target '{m.targ}' is not a single character.")


def check_char_integrity_in_sequences(
    dataset: Dataset,
    invalid_data_padding=True,
) -> tuple[Dataset, list[int]]:
    """Check if all elements in each sequence from a sequence list are in the alphabet.

    Args:
        dataset : Dataset to check
        replace :
            If True (default), replace all characters that are not in the alphabet with the padding character
            If False, raise an error if a character is not in the alphabet

    Returns:
        Dataset with replaced characters (if any false one was found)
            and the list of the rows without errors
    """
    seq_alphabet_dict = {
        "aa_seq": valid_aa_alphabet,
        "dna_seq": valid_dna_alphabet,
        "smiles_seq": valid_smiles_alphabet,
    }

    rows_with_errors = []  # FIXME Use dataset.map() instead of pandas
    df = dataset.to_pandas()
    for seq_col, alphabet in seq_alphabet_dict.items():
        if seq_col in df.columns:
            for id, sequence in enumerate(df[seq_col]):
                new_sequence, error = _check_sequence(
                    sequence, padding_value_enc, alphabet, invalid_data_padding
                )
                df.at[id, seq_col] = new_sequence
                if error:
                    rows_with_errors.append(id)

    rows_without_errors = set(range(len(df))) - set(rows_with_errors)

    return Dataset.from_pandas(df), list(rows_without_errors)


def _check_sequence(
    sequence: str,
    padding_char: str,
    alphabet: list[str],
    invalid_data_padding: bool = True,
):
    """Check if all elements in a sequence are in the alphabet.
    Args:
        sequence : Sequence to check
        alphabet : List of characters that are allowed in the sequences
        replace : If True, replace all characters that are not in the alphabet with the padding character
        padding_char: Character to replace the false ones with
            if False, raise an error if a character is not in the alphabet
    Returns:
        sequence : Sequence with replaced characters (if any false one was found)
        rows_with_errors : Boolean indicating if there were errors in the sequence
    """
    rows_with_errors = False

    if sequence is None:
        return sequence, False

    new_sequence = list(sequence)
    for pos, char in enumerate(sequence):
        if char not in [padding_char] + alphabet:
            if invalid_data_padding:
                print(f"{char} found in sequence {sequence}")
                rows_with_errors = True
                new_sequence[pos] = padding_char
            else:
                raise ValueError(
                    f"Found '{char}' -> not in alphabet {alphabet} - sequence: {sequence}"
                )
    return "".join(new_sequence), rows_with_errors

""" Module for encoding specifications.
This module contains the encoding specifications for the encodings that are provided by the package.

The encodings are specified as a list `encodings` containing `EncodingSpec` or `SklEncodingSpec` objects.

`EncodingSpec` objects contain the following attributes:
    - `provides` (List): The names of the columns that the encoding will provide
    - `requires` (List): The names of the columns that the encoding requires
    - `func`: A function that takes a dataset and returns a dictionary of columns that the encoding provides

SklEncodingSpec objects are slightly different and contain the following attributes:
    - `provides` (str): The name of the column that the encoding will provide
    - `requires` (str): The name of the column that the encoding requires
    - `func`: A sklearn Transformer (sklearn.TransformerMixin) object that will be fit and transformed on the dataset

The `encodings` list contains all the encodings that are provided by the package and is used by 
the `add_encodings` function, which takes a dataset or a dataset dictionary and adds the specified encodings to it.
Encodings are applied in a specific order based on their dependencies.

`provided_encodings` is a list of all the encodings that are provided by the package.
"""

from typing import Callable
from functools import partial
from ..encoding import (
    EncodingSpec,
    SklEncodingSpec,
    NGramFeat,
    Ankh,
    ESM,
    SeqStrOneHot,
    SeqStrLen,
    unirep,
    translate_dna_to_aa_seq,
    return_prob_feat,
)
from ..encoding import transforms_graph as tg
from .alphabets import (
    dna_alphabet,
    aa_alphabet,
    valid_smiles_alphabet,
    padding_value_enc,
)


encodings: list[EncodingSpec] = [
    # EncodingSpec(["list", "of", "provided", "encodings"], ["list_of","required_encodings"], function_taking_dataset_and_returning_dict),
    EncodingSpec(["atm_count", "bnd_count"], ["bnd_idcs"], tg.bnd_count_atm_count),
    EncodingSpec(["atm_adj"], ["atm_count", "bnd_count"], tg.atm_adj),
    EncodingSpec(["atm_bnd_incid"], ["atm_count", "bnd_count"], tg.atm_bnd_incid),
    EncodingSpec(
        ["atm_retprob100"],  # shape: (atm_count, atm_count)
        ["atm_adj"],  # shape: (2, atm_count)
        partial(
            return_prob_feat,
            100,
        ),
    ),
    EncodingSpec(["aa_unirep_1900", "aa_unirep_final"], ["aa_seq"], unirep),
    EncodingSpec(["aa_seq"], ["dna_seq"], translate_dna_to_aa_seq),
    EncodingSpec(
        ["aa_ankh_avg"],
        ["aa_ankh_base"],
        lambda df: {
            "aa_ankh_avg": df.with_format("numpy")["aa_ankh_base"].mean(axis=1).tolist()
        },
    ),
    EncodingSpec(
        ["aa_esm2_avg"],
        ["aa_esm2_t6_8M"],
        lambda df: {
            "aa_esm2_avg": df.with_format("numpy")["aa_esm2_t6_8M"]
            .mean(axis=1)
            .tolist()
        },
    ),
    EncodingSpec(
        ["aa_1hot"],
        ["aa_seq"],
        SeqStrOneHot("aa_1hot", "aa_seq", padding_value_enc, aa_alphabet),
    ),
    EncodingSpec(
        ["dna_1hot"],
        ["dna_seq"],
        SeqStrOneHot("dna_1hot", "dna_seq", padding_value_enc, dna_alphabet),
    ),
    EncodingSpec(
        ["smiles_1hot"],
        ["smiles_seq"],
        SeqStrOneHot(
            "smiles_1hot", "smiles_seq", padding_value_enc, valid_smiles_alphabet
        ),
    ),
    # Deprecated API below, only allows a single provided and single required encoding
    # SklEncodingSpec("provided", "required", sklearn.TransformerMixin),
    # This is not too much of an issue
    SklEncodingSpec("aa_len", "aa_seq", SeqStrLen()),  # aa_len shape: (1,)
    SklEncodingSpec("aa_1gram", "aa_seq", NGramFeat(1, aa_alphabet)),
    SklEncodingSpec("aa_ankh_base", "aa_seq", Ankh()),
    SklEncodingSpec("aa_esm2_t6_8M", "aa_seq", ESM()),
    SklEncodingSpec("dna_len", "dna_seq", SeqStrLen()),
]


provided_encodings = [p for e in encodings for p in e.provides]


def find_function_order(
    encoders: list[EncodingSpec],
    provided_encodings: list[str],
    required_encodings: list[str],
    satisfy_all: bool = True,
) -> list[Callable]:
    """
    Find the order in which to call encoding functions such that the required encodings can be computed.
    If the requirements are not satisfiable, throw an exception.

    Args:
        encoders (List[EncodingSpec]): List of encoding specifications.
        provided_encodings (List[str]): List of globally provided encodings.
        required_encodings (List[str]): List of required encodings.
        satisfy_all (bool, optional): If True, all encodings in `required_encodings` must be satisfied.
            If False, satisfy those that are satisfiable. Defaults to True.

    Returns:
        List[Callable]: A list of encoder functions in the order they should be called to satisfy the requirements.

    Raises:
        ValueError: If the requirements cannot be satisfied.

    Example:
        >>> from pedata.encoding import base
        >>> f1 = lambda x: x
        >>> f2 = lambda x: None
        >>> f3 = lambda x: 1
        >>> encoders = [
        ...     base.EncodingSpec(["aa_len"], ["aa_seq"], f1),
        ...     base.EncodingSpec(["aa_1hot"], ["aa_seq"], f2),
        ...     base.EncodingSpec(["dna_len"], ["dna_seq"], f3)
        ... ]
        >>> provided_encodings = ["aa_seq"]
        >>> required_encodings = ["aa_len"]
        >>> satisfy_all = True
        >>> result = find_function_order(encoders, provided_encodings, required_encodings, satisfy_all)
        >>> print(result) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [EncodingSpec(provides=['aa_len'], requires=['aa_seq'], ...)]
    """

    encoding_functions = {}  # Stores encoding functions for each encoder
    encodings_provided = {}  # Stores all avaialble encodings
    encodings_required = {}  # Stores all required/needed encodings
    encoding_specs = {}  # Stores all encoders

    if not satisfy_all:
        encodings = required_encodings
        required_encodings = []
        for encoding in encodings:
            # Skip encodings that require 'bnd_idcs' when it's not provided
            if (
                encoding.startswith("atm") or encoding.startswith("bnd")
            ) and "bnd_idcs" not in provided_encodings:
                continue

            # Skip encodings that require 'smiles_seq' when it's not provided
            if (
                encoding.startswith("smiles")
            ) and "smiles_seq" not in provided_encodings:
                continue

            if any(
                provided_enc.startswith("aa") for provided_enc in provided_encodings
            ):
                # If any Amino Acid encoding exists in provided_encodings, exclude DNA encodings
                if not encoding.startswith("dna"):
                    required_encodings.append(encoding)

            elif any(
                provided_enc.startswith("dna") for provided_enc in provided_encodings
            ):
                # If any DNA encoding exists in provided_encodings, exclude AA encodings
                if not encoding.startswith("aa"):
                    required_encodings.append(encoding)

    # Collect all necessary details from encoders
    for encoder in encoders:
        for encoding in encoder.provides:
            encoding_functions[encoding] = encoder.func
        encodings_provided[encoder.func] = encoder.provides
        encodings_required[encoder.func] = encoder.requires
        encoding_specs[encoder.func] = encoder

    # Topological sort using depth-first search
    sorted_list = []  # Store sorted list of functions
    checked = set()  # Stores encoding functions that have already been sorted

    def check_encoding(required_encodings: list[str]) -> None:
        """
        Check the providers of the required encodings.

        Args:
            required_encodings (List[str]): A list of required encodings.

        Raises:
            ValueError: If the requirements cannot be satisfied.
        """

        for encoding in required_encodings:
            # Check if encoding doesn't already exist
            if encoding not in provided_encodings:
                if satisfy_all and encoding not in encoding_functions:
                    raise ValueError(
                        f"Requirements cannot be satisfied. Lacking function for encoding: {encoding}"
                    )

                check_function(encoding_functions[encoding])

    def check_function(function: Callable):
        """
        Check if all function's encodings meet their requirements.

        Args:
            func (Callable): The function to check.
        """

        if function not in checked:
            checked.add(function)
            check_encoding(encodings_required[function])
            provided_encodings.extend(encodings_provided[function])
            sorted_list.append(function)

    check_encoding(required_encodings)

    if satisfy_all:
        # Check if all required encodings are in the sorted_list
        unsatisfied_requirements = set(required_encodings) - set(provided_encodings)

        # Throw an exception if there are unsatisfied requirements
        if unsatisfied_requirements:
            raise ValueError(
                f"Requirements cannot be satisfied: {unsatisfied_requirements}."
            )

    return [encoding_specs[f] for f in sorted_list]

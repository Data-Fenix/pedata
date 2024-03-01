# system
import os
import sys
from pathlib import Path

# maths
import jax.numpy as np  # FIXME can we use only numpy or is jax.numpy fully necessary here?
import numpy as onp

# peadata
from .encoding import alphabets

path_simil = (
    Path(os.path.realpath(os.path.join(os.path.dirname(__loader__.path), ".")))
    / "static"
    / "similarity_matrices"
)


def data_exists(filename: str) -> bool:
    """Check if a file exists.

    Args:
        filename (str): name of the file

    Returns:
        bool: True if file exists

    """
    return os.path.isfile(filename)


def load_similarity(
    alphabet_type: str,
    similarity_name: str | list[str],
    replace_existing: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Load similarity matrices.
        Loads similarity matrices based on the specified alphabet type and similarity names. It provides the capability to load
        multiple similarity matrices simultaneously and preprocesses them into usable similarity matrices. The function also supports
        optional caching of the loaded matrices for improved performance in subsequent operations.

    Args:
        alphabet_type: Specifies the type of alphabet used in the similarity calculation. It can be either "aa" for amino acids or "dna" for DNA sequences.
        similarity_name: The name or list of names of the similarity matrix/matrixes to be loaded.
        replace_existing: Determines if an existing matrix should be overwritten. Defaults to False.

    Returns:
        A tuple containing the alphabet used for the similarity calculation and the preprocessed similarity matrix.

    Example:
    >>> similarity_names = ['name1', 'name2']
    >>> alphabet, similarity_matrix = load_similarity('aa', similarity_names)
    >>> print(similarity_matrix)

    Raises:
        ValueError: If the specified alphabet type is invalid
        ValueError: If the similarity matrix dimensions are not valid. FIXME not tested
        ValueError: If the similarity matrix contains superfluous entries.
        Exception: If the similarity matrix is missing entries. FIXME not tested

    """

    # check which alphabet to use
    if alphabet_type == "aa":
        alph = onp.array(alphabets.aa_alphabet)
    elif alphabet_type == "dna":
        alph = onp.array(alphabets.dna_alphabet)
    else:
        raise ValueError(f"Invalid alphabet type: {alphabet_type}")

    rval = []  # Stores preprocessed similarity matrix as return value

    # Ensure similarity_name is a list
    if isinstance(similarity_name, str):
        similarity_name = [similarity_name]

    for s in similarity_name:
        # Prepare file paths
        similarity_filename, file_extension = alphabet_type + "_" + s, ".txt"
        output_file_path = os.path.join(
            path_simil, f"{similarity_filename}_ordered.csv"
        )

        if data_exists(output_file_path) and not replace_existing:
            # Load the similarity matrix from cache
            print(
                f"\n--- Existing disk cache ---\nFile: {output_file_path}\nStatus: Existing file will not be replaced\n---\n",
                file=sys.stderr,
            )
            rval.append(onp.loadtxt(output_file_path, delimiter=","))

        else:
            # Open the similarity matrix file and read its lines
            with open(
                os.path.join(path_simil, similarity_filename + file_extension)
            ) as matrix_file:
                lines = matrix_file.readlines()

            header = None  # Variable to store the header of the similarity matrix
            col_header = []  # List to store the column header of the similarity matrix
            similarity_matrix = []  # List to store the similarity matrix entries

            for idx, row in enumerate(lines):
                # Skip commented lines and empty lines
                if row[0] == "#" or len(row) == 0:
                    continue

                # Strip leading and trailing whitespace from the row
                row = row.strip()

                # Split the row into individual entries
                entries = row.split()

                if header is None:
                    # First non-comment and non-empty line represents the header
                    header = entries
                    continue

                else:
                    # The first entry in each subsequent line is the column header
                    col_header.append(entries.pop(0))

                    # Convert the remaining entries to floats and append them to the similarity matrix
                    similarity_matrix.append(list(map(float, entries)))

            # Convert the header and column header to numpy arrays
            header, col_header = onp.array(header), onp.array(col_header)

            # Convert the similarity matrix to a jax numpy array
            similarity_matrix = np.array(similarity_matrix)

            # Check the dimensions and consistency of the matrix
            if not np.all(header == col_header):  # FIXME: missing test
                raise ValueError(
                    "Inconsistent header and column header in the similarity matrix: "
                    "The values in the header and column header do not match."
                )

            if (
                len(header) != similarity_matrix.shape[0]
                or similarity_matrix.shape[0] != similarity_matrix.shape[1]
            ):
                raise ValueError("Dimensions of the similarity matrix are not valid.")

            # ?? Replace the missing value placeholder in the header if present
            if header[-1] == "*":
                header[-1] = alphabets.stop_codon_enc

            # Check for superfluous entries in the similarity matrix
            superfluous_entries = set(header).difference(alph)
            if len(superfluous_entries) > 0:
                print(
                    f"Similarity matrix contains superfluous entries {superfluous_entries}"
                )

            # Check for missing entries in the similarity matrix
            missing_entries = set(alph).difference(header)
            if len(missing_entries) != 0:  # FIXME: missing test
                raise Exception(f"Similarity matrix doesn't contain {missing_entries}")

            # Reorder the similarity matrix based on the alphabet order
            reorder = np.argmax(header[:, None] == alph[None, :], 0)

            # Append the reordered similarity matrix to the result list
            rval.append(similarity_matrix[reorder, :][:, reorder])

            # Save the reordered similarity matrix to disk for future use
            onp.savetxt(
                output_file_path,
                rval[-1],
                delimiter=",",
                header=", ".join(alph),
                fmt="%.2f",
            )

    return (
        alph,
        onp.array(rval).squeeze(),
    )  # Return the alphabet and the preprocessed similarity matrix

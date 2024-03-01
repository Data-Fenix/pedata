""" Module for adding encodings to datasets """

# datasets
import datasets as ds

# ml
from sklearn.base import TransformerMixin

# peadata
from .encoding_specs import encodings, provided_encodings, find_function_order


def add_encodings(
    dataset_dict: ds.DatasetDict | ds.Dataset,
    needed: list[str] | set[str] = None,
) -> ds.DatasetDict | ds.Dataset:
    """Add encodings to a single Dataset or Datasets in a dataset dictionary.

    This function takes a dataset dictionary or a single dataset and adds the specified encodings to it.
    Encodings are applied in a specific order based on their dependencies.

    Args:
        dataset_dict: Dataset or dataset dictionary to which encodings should be added
        needed: List or set of encodings to be added. Defaults to None.

    Returns:
        Dataset or dataset dictionary with new encodings added

    Raises:
        TypeError: If the input is not a valid dataset or dictionary of datasets.
        TypeError: If the `needed` parameter is not a list or valid encodings.

    Example:
        >>> needed = ["aa_len", "aa_1gram", "aa_1hot"]
        >>> dataset = ds.Dataset.from_dict(
        ...     {
        ...         "aa_mut": ["wildtype", "L2A"],
        ...         "aa_seq": ["MLGTK", "MAGTK"],
        ...         "target foo": [1, 2],
        ...     }
        ... )
        >>> encoded = add_encodings(dataset, needed)
        >>> print(encoded) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Dataset({
            features: ['aa_mut', 'aa_seq', 'target foo', 'aa_len', 'aa_1gram', 'aa_1hot'],
            num_rows: 2
        })
    """

    # do not include by default because they are very long to be computed
    #   "aa_ankh_base", "aa_esm2_t6_8M", "aa_esm2_avg", "aa_ankh_avg", "aa_unirep_1900", "atm_retprob100"
    # FIXME: add this above in the encoding list as fast_compute = False
    none_default_encodings = [
        "aa_ankh_base",
        "aa_ankh_avg",
        "aa_esm2_avg",
        "aa_esm2_t6_8M",
        "atm_retprob100",
    ]
    # If `dataset_dict` is a dictionary, iterate over each dataset and recursively call `add_encodings`
    if isinstance(dataset_dict, ds.DatasetDict):
        for name, dataset in dataset_dict.items():
            dataset_dict[name] = add_encodings(dataset, needed)

        return dataset_dict

    dataset = dataset_dict
    if isinstance(dataset, ds.Dataset):
        require_all = True  # Will help determine how strict encodings should be

        # If no encoding was provided, apply only encodings that are satisfiable by changing "require_all" to False
        if needed is None:
            needed = list(set(provided_encodings) - set(none_default_encodings))
            require_all = False

        if not isinstance(needed, list):
            raise TypeError("Input a valid list of encodings")

        # Determine the order in which encodings should be applied based on their dependencies

        func_order = find_function_order(
            encoders=encodings,
            provided_encodings=list(dataset.features.keys()),
            required_encodings=needed,
            satisfy_all=require_all,
        )

        # Apply the encoding functions and add the resulting columns to the dataset
        for enc in func_order:
            if isinstance(enc.func, TransformerMixin):
                if len(enc.provides) != 1:
                    raise Exception(
                        "Only single column encodings supported when using the old TransformerMixin interface"
                    )

                # Apply encoding using the `map_func` if available
                if hasattr(enc.func, "map_func"):
                    dataset = dataset.map(
                        lambda x: {enc.provides[0]: enc.func.map_func(x)},
                        writer_batch_size=100,
                        batch_size=100,
                        batched=True,
                    )
                else:
                    # Fit and transform the dataset using the encoding function
                    enc.func.fit(dataset)
                    val = enc.func.transform(dataset)
                    dataset = dataset.add_column(enc.provides[0], val)

            else:
                val = enc.func(dataset)

                for prov in enc.provides:
                    if prov not in val:
                        assert f"Encoding {enc} did not provide {prov} unlike specified"

                for k in val:
                    # if the column already exists - it might happen when multiple encodings are returned together (e.g. unirep)
                    if k in list(dataset.features.keys()):
                        dataset = dataset.remove_columns(
                            k
                        )  # TODO: write a test for this
                    # adds the column
                    dataset = dataset.add_column(k, val[k])

    else:
        raise TypeError("Invalid input type. Expected Dataset or a dataset dictionary.")

    return dataset

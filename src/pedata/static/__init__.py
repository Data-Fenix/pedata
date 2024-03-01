from .example_data.data import (
    RegressionToyDataset,
    ClassificationToyDataset,
    dataset_dict_regression,
    dna_example_1_missing_val,
    dna_example_1_no_missing_val,
    dna_example_1_missing_target,
    aa_example_0_missing_val,
    aa_example_0_no_missing_val,
    aa_example_0_missing_target,
    aa_example_0_invalid_alphabet,
)

from .example_data.dataset_fixtures import (
    regr_dataset_train,
    regr_dataset_test,
    regr_dataset,
    regr_dataset_splits,
    regr_dataset_as_1split_dict,
    regr_dna_dataset_train,
    regr_dna_dataset_test,
    regr_dna_dataset,
    regr_dna_dataset_splits,
    regr_dna_dataset_as_1split_dict,
)

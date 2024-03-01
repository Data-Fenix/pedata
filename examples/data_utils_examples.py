from pedata.preprocessing.tag_finder import TagFinder
from pedata.hfhub_tools import get_dataset

# Example of applying TagFinder to ProteineaSolubility
tf = TagFinder()
ds = get_dataset("Exazyme/ProteineaSolubility", as_DatasetDict=False)
no_tag_ds = tf.clean_dataset(dataset=ds, seq_col_name="aa_seq", num_proc=1)

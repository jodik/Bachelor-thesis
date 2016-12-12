from Programming.DataScripts import data_normalization
import Programming.DataScripts.data_process
from Programming.DataScripts import data_reader


def get_new_data_sets(permutation_index):
    full_data_set = data_reader.read_data()
    data_sets = Programming.DataScripts.data_process.process(full_data_set, permutation_index)
    data_sets = data_normalization.normalize_data_sets(data_sets)
    return data_sets

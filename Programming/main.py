import numpy as np

from Programming.TensorFlow import convolutional_network
from Programming.TensorFlow import convolutional_network_edges
from Programming.TensorFlow import pca_svm
import Programming.configuration as conf
from Programming.DataScripts import data_normalization
from Programming.DataScripts import data_reader
from Programming.DataScripts import data_process
from Programming.HelperScripts import helper


def compute(permutation_index):
    full_data_set = data_reader.read_data()
    data_sets = data_process.process(full_data_set, permutation_index)
    data_sets = data_normalization.normalize_data_sets(data_sets)

    #return convolutional_network_edges.compute(data_sets)
    #return pca_svm.compute(data_sets)
    return convolutional_network.compute(data_sets)


def main():
    np.random.seed(conf.SEED)
    if conf.FULL_CROSS_VALIDATION:
        error = 0
        confusion_matrix_across_all_iterations = np.zeros((len(conf.DATA_TYPES_USED), len(conf.DATA_TYPES_USED)), dtype=int)
        for i in range(conf.CROSS_VALIDATION_ITERATIONS):
            print('\nCOMPUTE %d. CROSSVALIDATION:\n' % (i+1))
            test_error, confusion_matrix = compute(i)
            error += test_error
            confusion_matrix_across_all_iterations += confusion_matrix

        print('\n\n Full Cross Validation results:\n')
        helper.write_test_stats(confusion_matrix_across_all_iterations, error / conf.CROSS_VALIDATION_ITERATIONS)
    else:
        compute(conf.PERMUTATION_INDEX)

if __name__ == '__main__':
  main()
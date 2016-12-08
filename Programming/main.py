import numpy as np

import Programming.configuration as conf
from Programming.DataScripts import data_normalization
from Programming.DataScripts import data_process
from Programming.DataScripts import data_reader
from Programming.HelperScripts import helper
from Programming.Learning.CNN import cnn_edges
from Programming.Learning.CNN.cnn_default import CNNDefault
from Programming.Learning.CNN.cnn_edges import CNNEdges


def compute(permutation_index):
    full_data_set = data_reader.read_data()
    data_sets = data_process.process(full_data_set, permutation_index)
    data_sets = data_normalization.normalize_data_sets(data_sets)

    cnn_run = CNNDefault(data_sets)
    #cnn_run = CNNEdges(data_sets)
    return cnn_run.run()
    #return pca_svm.compute(data_sets)


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
        helper.write_eval_stats(confusion_matrix_across_all_iterations, error / conf.CROSS_VALIDATION_ITERATIONS)
    else:
        compute(conf.PERMUTATION_INDEX)

if __name__ == '__main__':
    main()

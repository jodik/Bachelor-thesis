import numpy as np
import Programming.configuration as conf
from Programming.DataScripts import data_normalization
from Programming.DataScripts import data_process
from Programming.DataScripts import data_reader
from Programming.HelperScripts import helper
from Programming.HelperScripts import redirect_output_stream
from Programming.HelperScripts.what_to_run import WhatToRun


WHAT_TO_RUN = WhatToRun.cnn_default


def compute(permutation_index):
    full_data_set = data_reader.read_data()
    data_sets = data_process.process(full_data_set, permutation_index)
    data_sets = data_normalization.normalize_data_sets(data_sets)

    run_object = WHAT_TO_RUN(data_sets)
    return run_object.run()


def main():
    np.random.seed(conf.SEED)
    if conf.WRITE_TO_FILE:
        redirect_output_stream.change_output_stream(WHAT_TO_RUN)
    if conf.FULL_CROSS_VALIDATION:
        error = 0
        confusion_matrix_across_all_iterations = np.zeros((conf.NUM_LABELS, conf.NUM_LABELS), dtype=int)
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

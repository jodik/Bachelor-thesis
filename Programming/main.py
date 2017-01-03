import numpy as np
import Programming.configuration as conf
from Programming.HelperScripts import helper
import Programming.HelperScripts.redirect_output_stream as redirect_output_stream
from Programming.HelperScripts.what_to_run import WhatToRun
from DataScripts.data_sets_retrieval import get_new_data_sets
from Programming.HelperScripts.time_calculator import TimeCalculator

WHAT_TO_RUN = WhatToRun.pca_svm


def compute(permutation_index):
    data_sets = get_new_data_sets(permutation_index)

    run_object = WHAT_TO_RUN(data_sets)
    return run_object.run()


def main():
    np.random.seed(conf.SEED)
    if conf.WRITE_TO_FILE:
        redirect_output_stream.change_output_stream(WHAT_TO_RUN)
    if conf.FULL_CROSS_VALIDATION:
        time_logger = TimeCalculator("Full Cross-Validation")
        time_logger.show("Started")
        error = 0
        confusion_matrix_across_all_iterations = np.zeros((conf.NUM_LABELS, conf.NUM_LABELS), dtype=int)
        for i in range(conf.CROSS_VALIDATION_ITERATIONS):
            print('\nCOMPUTE %d. CROSSVALIDATION:\n' % (i+1))
            test_error, confusion_matrix = compute(i)
            error += test_error
            confusion_matrix_across_all_iterations += confusion_matrix

        print('\n\n Full Cross Validation results:\n')
        helper.write_eval_stats(confusion_matrix_across_all_iterations, error / conf.CROSS_VALIDATION_ITERATIONS)
        time_logger.show("Finished")
    else:
        compute(conf.PERMUTATION_INDEX)

if __name__ == '__main__':
    main()

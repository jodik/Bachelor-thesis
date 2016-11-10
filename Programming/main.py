import numpy
import Programming.TensorFlow.configuration as conf
import Programming.TensorFlow.ConvolutionalNetwork.myDatasetCon as mds
import Programming.TensorFlow.ConvolutionalNetwork.myConvolutional as conv


def compute(permutation_index):
    myset = mds.read_data_sets(permutation_index)
    return conv.compute(myset)


def main():
    if conf.FULL_CROSS_VALIDATION:
        error = 0
        confusion_matrix_across_all_iterations = numpy.zeros((len(conf.DATA_TYPES_USED), len(conf.DATA_TYPES_USED)), dtype=int)
        for i in range(conf.CROSS_VALIDATION_ITERATIONS):
            print('\nCOMPUTE %d. CROSSVALIDATION:\n' % (i+1))
            test_error, confusion_matrix = compute(i)
            error += test_error
            confusion_matrix_across_all_iterations += confusion_matrix

        print('\n\n Full Cross Validation results:\n')
        writeTestStats(confusion_matrix_across_all_iterations, error / conf.CROSS_VALIDATION_ITERATIONS)
    else:
        compute(conf.PERMUTATION_INDEX)

if __name__ == '__main__':
  main()
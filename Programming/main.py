import numpy
import Programming.TensorFlow.configuration as conf
import Programming.TensorFlow.ConvolutionalNetwork.myConvolutional as conv
from Programming.DataScripts import data_reader
from Programming.DataScripts import data_normalization


def compute(permutation_index):
    data_sets = data_reader.read_datasets(permutation_index)

    data_normalization.normalize(data_sets)

    train_perm, validation_perm, test_perm = map(lambda x: data_normalization.equalCountsPerms(x),
                                                 data_sets.getLabelSets())
    data_sets.train.apply_permutation(train_perm)
    data_sets.validation.apply_permutation(validation_perm)
    data_sets.test.apply_permutation(test_perm)

    return conv.compute(data_sets)


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
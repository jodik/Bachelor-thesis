from texttable import Texttable
import Programming.configuration as conf


def get_label_index(path, data_types):
    for i in range(len(data_types)):
        if data_types[i] in path:
            return i
    raise ValueError('Bad path.')


def write_confusion_matrix(matrix):
    table = Texttable()

    data = []
    for i in range(len(conf.DATA_TYPES_USED)):
        tmp = [conf.DATA_TYPES_USED[i]]
        count = 0
        for num in matrix[i]:
            tmp.append(num)
            count += num
        tmp.append('{0:.1f}%'.format(matrix[i][i] * 100.0 / count))
        data.append(tmp)

    cols = [''] + conf.DATA_TYPES_USED + ['Predicted']
    cols = map(lambda x: x[:4], cols)
    table.add_rows([cols] + data)
    print table.draw()


def write_test_stats(test_confusion_matrix, test_error):
    percentage_each_category_same_value = 0.0
    for i in range(len(test_confusion_matrix)):
        percentage_each_category_same_value += test_confusion_matrix[i, i] / sum(test_confusion_matrix[i])
    percentage_each_category_same_value /= len(test_confusion_matrix)
    percentage_each_category_same_value *= 100
    print('Test error: %.1f%%' % test_error)
    print('Test error, each category same value: %.1f%%' % (100 - percentage_each_category_same_value))
    write_confusion_matrix(test_confusion_matrix)

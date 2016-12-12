from texttable import Texttable
from os import listdir
from os.path import join
import Programming.configuration as conf


def get_label_index(path, data_types):
    for i in range(len(data_types)):
        if data_types[i] in path:
            return i
    raise ValueError('Bad path.')


def write_confusion_matrix(matrix):
    table = Texttable()

    data = []
    for i in range(conf.NUM_LABELS):
        tmp = [conf.DATA_TYPES_USED[i]]
        if i == 0 and conf.SIMPLIFIED_CATEGORIES:
            tmp = ['Bottles']
        count = 0
        for num in matrix[i]:
            tmp.append(num)
            count += num
        tmp.append('{0:.1f}%'.format(matrix[i][i] * 100.0 / count))
        data.append(tmp)

    cols = [''] + conf.DATA_TYPES_USED + ['Predicted']
    if conf.SIMPLIFIED_CATEGORIES:
        cols[1] = 'Bottles'
        cols = cols[0:4] + [cols[-1]]
    cols = map(lambda x: x[:4], cols)
    table.add_rows([cols] + data)
    print table.draw()


def write_eval_stats(eval_confusion_matrix, eval_error, test_data = False):
    type_of_result = "Test" if test_data else "Validation"
    print(type_of_result + ' error: %.1f%%' % eval_error)
    write_confusion_matrix(eval_confusion_matrix)


def main_directory():
    current_directory = './'
    current_folders = listdir(current_directory)
    while 'Programming' not in current_folders:
        current_directory = join(current_directory, '..')
        current_folders = listdir(current_directory)
    return current_directory + '/'


def unpack_file(path):
    return open(main_directory() + path, 'r').read().splitlines()


def write_line():
    print '--------------------------------'

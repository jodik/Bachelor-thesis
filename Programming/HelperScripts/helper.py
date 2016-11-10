from texttable import Texttable

import Programming.configuration as conf


def getLabelIndex(path, data_types):
     for i in range(len(data_types)):
         if data_types[i] in path:
             return i
     raise ValueError('Bad path.')

def writeConfusionMatrix(matrix):
    table = Texttable()

    data = []
    for i in range(len(conf.DATA_TYPES_USED)):
        tmp = [conf.DATA_TYPES_USED[i]]
        count = 0
        for num in matrix[i]:
            tmp.append(num)
            count += num
        tmp.append('{0:.1f}%'.format(matrix[i][i]*100.0/count))
        data.append(tmp)

    cols = [''] + conf.DATA_TYPES_USED + ['Predicted']
    cols = map(lambda x: x[:4], cols)
    table.add_rows([cols] + data)
    print table.draw()

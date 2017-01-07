import os
from texttable import Texttable


def add(lines):
    val_err = -1
    c = -1
    gamma = -1
    kernel = -1
    com = -1
    simplified = False
    hard_difficulty = False
    border = False
    extended_dataset = False
    for i in range(len(lines)):
        line = lines[i]
        if 'NUM_OF_COMPONENTS = ' in line:
            com = line.split('NUM_OF_COMPONENTS = ')[1]
        if 'PARAM_GRID = {\'C\': ' in line:
            c = line.split('PARAM_GRID = {\'C\': ')[1][:-1]
        if '\'gamma\': ' in line:
            gamma = line.split('\'gamma\': ')[1][:-1]
        if '\'kernel\': ' in line:
            kernel = line.split('\'kernel\': ')[1]
        if 'Full Cross Validation results:' in line:
            val_err = lines[i+2].split('Validation error: ')[1]
        if 'EXTENDED_DATASET = ' in line:
            extended_dataset = line.split('EXTENDED_DATASET = ')[1] == 'True'
        if 'BLACK_BORDER = ' in line:
            border = line.split('BLACK_BORDER = ')[1] == 'True'
        if 'HARD_DIFFICULTY = ' in line:
            hard_difficulty = line.split('HARD_DIFFICULTY = ')[1] == 'True'
        if 'SIMPLIFIED_CATEGORIES = ' in line:
            simplified = line.split('SIMPLIFIED_CATEGORIES = ')[1] == 'True'

    return (simplified, hard_difficulty, border, extended_dataset), val_err, com, c, gamma, kernel

out_index_folder = 'Learning/PCA_SVM/results/default/full_cv/'
index = int(open(out_index_folder + 'index.txt', 'r').readline()[:-1])

all = {}
for i in range(index):
    path = out_index_folder + 'out_'+str(i)+'.txt'
    if os.path.exists(path):
        params, val_err, com, c, gamma, kernel = add(open(path, 'r').read().splitlines())
        if val_err > 0:
            if params not in all:
                all[params] = []
            all[params].append([val_err, com, c, gamma, kernel, i])
for keys in all.keys():
    print('Simplified', keys[0], 'Hard difficulty', keys[1], 'Black border', keys[2], 'Extended dataset', keys[3])
    values = all[keys]
    values = sorted(values)
    table = Texttable()
    cols = ['Val Err', 'Compon.', 'C', 'gamma', 'kernel', 'Index']
    table.add_rows([cols] + values)
    print table.draw()


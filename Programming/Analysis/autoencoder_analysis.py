import os
from texttable import Texttable

def my_split(what, line, before):
    if what in line:
        return line.split(what)[1]
    else:
        return before

def add(lines):
    val_err = -1
    ep = -1
    fc1 = -1
    opt = -1
    simplified = False
    hard_difficulty = False
    border = False
    extended_dataset = False
    for i in range(len(lines)):
        line = lines[i]
        ep = my_split('NUMBER_OF_EPOCHS = ', line, ep)
        fc1 = my_split('NUMBER_OF_NEURONS_IN_HIDDEN_LAYER = ', line, fc1)
        opt = my_split('OPTIMIZER = ', line, opt)
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
        if 'Full Cross-Validation: Total time:' in line:
            time = line.split('Full Cross-Validation: Total time:')[1]


    return (simplified, hard_difficulty, border, extended_dataset), val_err, ep, fc1, opt

out_index_folder = 'Learning/Autoencoder/results/prediction/simple/full_cv/'
index = int(open(out_index_folder + 'index.txt', 'r').readline()[:-1])


def get_results():
    all = {}
    for i in range(index):
        path = out_index_folder + 'out_'+str(i)+'.txt'
        if os.path.exists(path):
            params, val_err, ep, fc1, opt = add(open(path, 'r').read().splitlines())
            if val_err > 0:
                if params not in all:
                    all[params] = []
                all[params].append([val_err, ep, fc1, opt, i])
    return all


def main():
    all = get_results()
    for keys in all.keys():
        print('Simplified', keys[0], 'Hard difficulty', keys[1], 'Black border', keys[2], 'Extended dataset', keys[3])
        values = all[keys]
        values = sorted(values)
        table = Texttable()
        cols = ['Val Err', 'Epochs', 'Fc1', 'Optimizer', 'Index']
        table.add_rows([cols] + values)
        print table.draw()

if __name__ == '__main__':
    main()

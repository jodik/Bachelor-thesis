import os
from texttable import Texttable


def my_split(what, line, before):
    if what in line:
        return line.split(what)[1]
    else:
        return before


def add(lines):
    val_err = -1
    lr = -1
    fc1 = -1
    dc = -1
    m = -1
    dp = -1
    cfs = -1
    cff = -1
    csf = -1
    cfd = -1
    csd = -1
    vf = -1
    scale = -1
    time = '-'
    simplified = False
    hard_difficulty = False
    border = False
    extended_dataset = False
    for i in range(len(lines)):
        line = lines[i]
        lr = my_split('BASE_LEARNING_RATE = ', line, lr)
        dc = my_split('DECAY_RATE = ', line, dc)
        fc1 = my_split('FC1_FEATURES = ', line, fc1)
        m = my_split('MOMENTUM = ', line, m)
        dp = my_split('DROPOUT_PROBABILITY = ', line, dp)
        cfs = my_split('CON_FIRST_STRIDE = ', line, cfs)
        cff = my_split('CONV_FIRST_FILTER_SIZE = ', line, cff)
        csf = my_split('CONV_SECOND_FILTER_SIZE = ', line, csf)
        cfd = my_split('CONV_FIRST_DEPTH = ', line, cfd)
        csd = my_split('CONV_SECOND_DEPTH = ', line, csd)
        vf = my_split('VALIDATION_FREQUENCY = ', line, vf)
        scale = my_split('SCALE = ', line, scale)
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

    return (simplified, hard_difficulty, border, extended_dataset, scale), [val_err, lr, dc, fc1, m, time, dp, cfs, cff, csf, cfd, csd, vf]

out_index_folder = 'Learning/CNN/results/default/full_cv/'
index = int(open(out_index_folder + 'index.txt', 'r').readline()[:-1])


def get_results():
    all = {}
    for i in range(index):
        path = out_index_folder + 'out_'+str(i)+'.txt'
        if os.path.exists(path):
            params, local_params = add(open(path, 'r').read().splitlines())
            if local_params[0] > 0:
                if params not in all:
                    all[params] = []
                all[params].append(local_params + [i])
    return all


def select(fr, indicies):
    res = []
    for y in fr:
        values_selected = []
        for x in indicies:
            values_selected += [y[x]]
        res += [values_selected]
    return res

def main():
    all = get_results()
    for keys in all.keys():
        print('Simplified', keys[0], 'Hard difficulty', keys[1], 'Black border', keys[2], 'Extended dataset', keys[3], 'SCALE', keys[4])
        values = all[keys]
        values = sorted(values)
        table = Texttable()
        print len(values[0])
        indices = [0, 6, 7, 8, 9, 10, 11, 12, 13]
        cols = ['ve', 'lr', 'dc', 'fc1', 'm', 't', 'dp', 'cfs', 'cff', 'csf', 'cfd', 'csd', 'vf', 'i']
        values = select(values, indices)
        cols = select([cols], indices)
        table.add_rows(cols + values)
        print table.draw()

if __name__ == '__main__':
    main()

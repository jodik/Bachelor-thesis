from texttable import Texttable
from Programming.Analysis import autoencoder_analysis, cnn_results_analysis

all = autoencoder_analysis.get_results()

parm = (False, True, True, True)
values = all[parm]
values = sorted(values)
#indices = [0, 3,4,5,6, 7, 10, 11, 13]
indices = [0, 1, 2, 4]
cols = ['Val Err', 'Epochs', 'Fc1', 'Optimizer', 'Index']
values = cnn_results_analysis.select(values, indices)
cols = cnn_results_analysis.select([cols], indices)
#cols = [cols]
cols[0].append('or i')

i=0
for v in values:
    v.append(i)
    i+=1

table = Texttable()
print len(values[0])
table.add_rows(cols + values)
print table.draw()
indicies = [0]
indicies = sorted(indicies)
values_selected = []
for x in indicies:
    values_selected += [values[x]]

values_selected = values[0:]

print('\\begin{center}')
print('\\begin{table}')
print('\\begin{tabular}{ | l | l | l | l | l | l | l | l | l | l |}')
print('\\hline')
for x in cols[0]:
    print x,' & ',
print '\\\\'
#print('Index & Components & C & Gamma & Kernel f. & Time & Val. error \\\\ ')
for tuple in values_selected:
    print('\\hline')
    print tuple[-2],
    for x in tuple[1:-2]:
        print '&',
        print x,
    print '&',
    print tuple[0][:-1] + '\\%',
    print '\\\\'


print('\\hline')
print('\end{tabular}')
print('\caption{Test}')
print('\label{tab:test}')
print('\end{table}')
print('\end{center}')

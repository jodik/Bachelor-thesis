from texttable import Texttable
from Programming.Analysis import pca_svm_results_analysis

all = pca_svm_results_analysis.get_results()

parm = (False, False, True, True)
values = all[parm]
values = sorted(values)

i=0
for v in values:
    v.append(i)
    i+=1

table = Texttable()
cols = ['Val Err', 'Compon.', 'C', 'gamma', 'kernel', 'Time', 'Index', 'Or. Ind']
table.add_rows([cols] + values)
print table.draw()
indicies = [0,14,22,29,38]
indicies = sorted(indicies)
values_selected = []
for x in indicies:
    values_selected += [values[x]]

values_selected = values[21:]

print('\\begin{center}')
print('\\begin{table}')
print('\\begin{tabular}{ | l | l | l | l | l | l | l |}')
print('\\hline')
print('Index & Components & C & Gamma & Kernel f. & Time & Val. error \\\\ ')
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

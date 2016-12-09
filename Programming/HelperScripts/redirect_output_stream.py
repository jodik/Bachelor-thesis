from Programming.HelperScripts.what_to_run import WhatToRun
import Programming.configuration as conf
import os
import sys


OUTPUT_PATHS = {WhatToRun.cnn_default: 'CNN/results/default/',
                WhatToRun.cnn_edges: 'CNN/results/edges/',
                WhatToRun.pca_svm: 'PCA_SVM/results/default/',
                WhatToRun.pca_svm_edges: 'PCA_SVM/results/edges/'}


def retrieve_output_folder(WHAT_TO_RUN):
    path = 'Learning/' + OUTPUT_PATHS[WHAT_TO_RUN]
    path = path + 'full_cv/' if conf.FULL_CROSS_VALIDATION else path + 'permutation_' + str(
        conf.PERMUTATION_INDEX) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def change_output_stream(WHAT_TO_RUN):
    path = retrieve_output_folder(WHAT_TO_RUN)
    out_index_file_name = path + 'index.txt'
    if not os.path.isfile(out_index_file_name):
        out_index_file = open(out_index_file_name, 'w')
        out_index_file.write('0\n')
        out_index_file.close()
    index = open(out_index_file_name, 'r').readline()[:-1]
    sys.stdout = open(path + "out_" + index + ".txt", 'w')


def update_file_index(WHAT_TO_RUN):
    path = retrieve_output_folder(WHAT_TO_RUN)
    out_index_file_name = path + 'index.txt'
    index = open(out_index_file_name, 'r').readline()[:-1]
    out_index_file = open(out_index_file_name, 'w')
    out_index_file.write(str(int(index) + 1) + '\n')
    out_index_file.close()

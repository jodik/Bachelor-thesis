import os
import sys
from Programming.HelperScripts.what_to_run import WhatToRun
import Programming.configuration as conf
from Programming.HelperScripts import helper


OUTPUT_PATHS = {WhatToRun.cnn_default: 'CNN/results/default/',
                WhatToRun.cnn_edges: 'CNN/results/edges/',
                WhatToRun.cnn_4_channels: 'CNN/results/4_channels/',
                WhatToRun.cnn_wide: 'CNN/results/wide/',
                WhatToRun.cnn_edges_deep_3: 'CNN/results/edges_deep_3/',
                WhatToRun.cnn_default_deep_3: 'CNN/results/default_deep_3/',
                WhatToRun.pca_svm: 'PCA_SVM/results/default/',
                WhatToRun.pca_svm_edges: 'PCA_SVM/results/edges/'}

CONFIGURATION_PATHS = {WhatToRun.cnn_default: 'CNN/configuration_default.py',
                       WhatToRun.cnn_edges: 'CNN/configuration_edges.py',
                       WhatToRun.cnn_4_channels: 'CNN/configuration_4_channels.py',
                       WhatToRun.cnn_wide: 'CNN/configuration_default.py',
                       WhatToRun.cnn_edges_deep_3: 'CNN/configuration_edges_deep_3.py',
                       WhatToRun.cnn_default_deep_3: 'CNN/configuration_default_deep_3.py',
                       WhatToRun.pca_svm: 'PCA_SVM/results/default/',
                       WhatToRun.pca_svm_edges: 'PCA_SVM/results/edges/'}


def retrieve_output_folder(WHAT_TO_RUN):
    path = 'Learning/' + OUTPUT_PATHS[WHAT_TO_RUN]
    path = path + 'full_cv/' if conf.FULL_CROSS_VALIDATION else path + 'permutation_' + str(
        conf.PERMUTATION_INDEX) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def log_config_files(WHAT_TO_RUN):
    print ('CONFIGURATION LOCAL:')
    conf_important_lines = helper.unpack_file('Programming/Learning/' + CONFIGURATION_PATHS[WHAT_TO_RUN])[2:22]
    for x in conf_important_lines:
        print(x)
    helper.write_line()
    print ('CONFIGURATION GLOBAL:')
    conf_important_lines = helper.unpack_file('Programming/configuration.py')[:11]
    for x in conf_important_lines:
        print(x)
    helper.write_line()


def update_file_index(out_index_file_name, current_index):
    out_index_file = open(out_index_file_name, 'w')
    out_index_file.write(str(int(current_index) + 1) + '\n')
    out_index_file.close()


def change_output_stream(WHAT_TO_RUN):
    path = retrieve_output_folder(WHAT_TO_RUN)
    out_index_file_name = path + 'index.txt'
    if not os.path.isfile(out_index_file_name):
        out_index_file = open(out_index_file_name, 'w')
        out_index_file.write('0\n')
        out_index_file.close()
    index = open(out_index_file_name, 'r').readline()[:-1]
    sys.stdout = open(path + "out_" + index + ".txt", 'w')
    log_config_files(WHAT_TO_RUN)
    update_file_index(out_index_file_name, index)



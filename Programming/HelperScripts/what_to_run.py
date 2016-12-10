from Programming.Learning.CNN.cnn_default import CNNDefault
from Programming.Learning.CNN.cnn_edges import CNNEdges
from Programming.Learning.CNN.cnn_4_channels import CNN4Channels
from enum import Enum
from Programming.Learning.PCA_SVM import pca_svm, pca_svm_edges


class WhatToRun(Enum):
    cnn_default = CNNDefault
    cnn_edges = CNNEdges
    cnn_4_channels = CNN4Channels
    pca_svm = pca_svm
    pca_svm_edges = pca_svm_edges

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val(*args, **kwargs)

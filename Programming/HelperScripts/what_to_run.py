from Programming.Learning.Autoencoder.autoencoder_deep import DeepAutoEncoder
from Programming.Learning.CNN.cnn_default import CNNDefault
from Programming.Learning.CNN.cnn_edges import CNNEdges
from Programming.Learning.CNN.cnn_res import CNNRes
from Programming.Learning.CNN.cnn_wide import CNNWide
from Programming.Learning.CNN.cnn_4_channels import CNN4Channels
from Programming.Learning.CNN.cnn_edges_deep_3 import CNNEdgesDeep3
from Programming.Learning.CNN.cnn_default_deep_3 import CNNDefaultDeep3
from enum import Enum
from Programming.Learning.PCA_SVM import pca_svm, pca_svm_edges
from Programming.Learning.Autoencoder.autoencoder_simple import SimpleAutoEncoder
from Programming.Learning.Autoencoder.classifier_deep import ClassifierDeepAutoEncoder
from Programming.Learning.Autoencoder.visualization_simple import VisualisationSimpleAutoEncoder
from Programming.Learning.Autoencoder.classifier_simple import ClassifierSimpleAutoEncoder


class WhatToRun(Enum):
    cnn_default = CNNDefault
    cnn_edges = CNNEdges
    cnn_4_channels = CNN4Channels
    cnn_wide = CNNWide
    cnn_edges_deep_3 = CNNEdgesDeep3
    cnn_default_deep_3 = CNNDefaultDeep3
    pca_svm = pca_svm.PCA_SVM
    pca_svm_edges = pca_svm_edges
    simple_autoencoder = SimpleAutoEncoder
    deep_autoencoder = DeepAutoEncoder
    visualisation_simple_autoencoder = VisualisationSimpleAutoEncoder
    simple_autoencoder_classifier = ClassifierSimpleAutoEncoder
    deep_autoencoder_classifier = ClassifierDeepAutoEncoder
    cnn_res = CNNRes

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val(*args, **kwargs)

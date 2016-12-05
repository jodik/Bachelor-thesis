from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from time import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import Programming.configuration as conf
from Programming.DataScripts import data_reader, data_process, data_normalization
import cv2

def plot_gallery(images, titles, h, w, depth, n_row=5, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)))
        #print(images[i].reshape((h, w))[50])
        #cv2.imshow('sd', images[i].reshape((h, w)))
        #cv2.waitKey(0)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def compute(data_sets):
    n_samples, h, w = data_sets.train.edge_descriptors.shape
    print(n_samples, h, w)
    data_sets.flatten()
    #cv2.imshow('sd', data_sets.train.edge_descriptors[0].reshape(h,w))
    #cv2.waitKey(0)
    n_components = 5
    pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True)
    pca.fit(data_sets.train.edge_descriptors)

    eigenfaces = pca.components_.reshape((n_components, h, w))

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    #plot_gallery(eigenfaces, eigenface_titles, h, w, 1, 4, 3)
    #plt.show()

    X_t_train = pca.transform(data_sets.train.edge_descriptors)
    X_t_test = pca.transform(data_sets.test.edge_descriptors)

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1, 3, 5, 8, 10],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    param_grid = {'C': [10],
                  'gamma': [0.0075],
                  'kernel': ['rbf']}
    clf = GridSearchCV(SVC(class_weight='balanced', random_state=conf.SEED), param_grid)
    clf = clf.fit(X_t_train, data_sets.train.labels.reshape(data_sets.train.labels.shape[0]))
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_t_test)
    print("done in %0.3fs" % (time() - t0))

    y_test = data_sets.test.labels.reshape(data_sets.test.labels.shape[0])
    print(classification_report(y_test, y_pred, target_names=conf.DATA_TYPES_USED))
    print(confusion_matrix(y_test, y_pred, labels=range(len(conf.DATA_TYPES_USED))))

    test_error, _, _, _ = precision_recall_fscore_support(y_test, y_pred,
                                                  average='macro')
    test_error *= 100
    test_error = 100 - test_error

    print ("%.1f%%" % test_error)
    return test_error, confusion_matrix(y_test, y_pred, labels=range(len(conf.DATA_TYPES_USED)))


if __name__ == '__main__':
  main()

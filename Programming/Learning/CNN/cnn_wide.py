import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import tensorflow as tf
from Programming.Learning.CNN.cnn_default import CNNDefault
from Programming.Learning.CNN.cnn_edges import CNNEdges
import Programming.configuration as conf
from Programming.DataScripts.data_sets_retrieval import get_new_data_sets


class CNNWide(object):
    def __init__(self, data_sets):
        self.data_sets_default = data_sets
        print(data_sets.train.edge_descriptors.shape)
        self.cnn_default = CNNDefault(data_sets)
        conf.update_simplified_categories(conf, True)
        self.data_sets_edges = get_new_data_sets(conf.PERMUTATION_INDEX)
        self.cnn_edges = CNNEdges(self.data_sets_edges)
        conf.update_simplified_categories(conf, False)
        print(self.data_sets_default.train.edge_descriptors.shape)

    def init_name(self):
        self.name = "CNN Wide"

    def run(self):
        self.cnn_default.run()
        conf.update_simplified_categories(conf, True)
        self.cnn_edges.run()
        self.fit_SVM()

    def fit_SVM(self):
        data = self.data_sets_default.train
        print (self.data_sets_default.train.labels[0:4])
        print (self.data_sets_edges.train.labels[0:4])
        train_data1 = self.cnn_default.sess.run(self.cnn_default.model.train_eval_prediction, feed_dict={self.cnn_default.model.train_eval_data_node: data.images})

        self.cnn_edges.model.train_eval_data_node = tf.placeholder(
            tf.float32,
            shape=(self.data_sets_default.train.labels.shape[0], self.cnn_edges.model.configuration_specific.IMAGE_HEIGHT,
                   self.cnn_edges.model.configuration_specific.IMAGE_WIDTH, self.cnn_edges.model.configuration_specific.NUM_CHANNELS))
        self.cnn_edges.model.train_eval_prediction = tf.nn.softmax(self.cnn_edges.model.create_train_eval_model())
        print(data.edge_descriptors.shape)

        train_data2 = self.cnn_edges.sess.run(self.cnn_edges.model.train_eval_prediction,
                                               feed_dict={self.cnn_edges.model.train_eval_data_node: data.edge_descriptors})
        train_data = np.concatenate([train_data1, train_data2], axis=1)

        param_grid = {'C': [10],
                      'gamma': [0.0075],
                      'kernel': ['rbf']}
        clf = GridSearchCV(SVC(class_weight='balanced', random_state=conf.SEED), param_grid)
        clf = clf.fit(train_data, self.data_sets_default.train.labels.reshape(self.data_sets_default.train.labels.shape[0]))

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        data = self.data_sets_default.validation
        val_data1 = self.cnn_default.sess.run(self.cnn_default.model.eval_prediction,
                                                feed_dict={self.cnn_default.model.eval_data_node: data.images})

        self.cnn_edges.model.eval_data_node = tf.placeholder(
            tf.float32,
            shape=(
            self.data_sets_default.validation.labels.shape[0], self.cnn_edges.model.configuration_specific.IMAGE_HEIGHT,
            self.cnn_edges.model.configuration_specific.IMAGE_WIDTH,
            self.cnn_edges.model.configuration_specific.NUM_CHANNELS))
        self.cnn_edges.model.eval_prediction = tf.nn.softmax(self.cnn_edges.model.create_eval_model())

        val_data2 = self.cnn_edges.sess.run(self.cnn_edges.model.eval_prediction,
                                                feed_dict={self.cnn_edges.model.eval_data_node: data.edge_descriptors})
        val_data = np.concatenate([val_data1, val_data2], axis=1)

        y_pred = clf.predict(val_data)

        y_test = self.data_sets_default.test.labels.reshape(self.data_sets_default.test.labels.shape[0])
        print(classification_report(y_test, y_pred, target_names=conf.DATA_TYPES_USED))
        print(confusion_matrix(y_test, y_pred, labels=range(len(conf.DATA_TYPES_USED))))

        test_error, _, _, _ = precision_recall_fscore_support(y_test, y_pred,
                                                              average='macro')
        test_error *= 100
        test_error = 100 - test_error

        return test_error, confusion_matrix(y_test, y_pred, labels=range(len(conf.DATA_TYPES_USED)))

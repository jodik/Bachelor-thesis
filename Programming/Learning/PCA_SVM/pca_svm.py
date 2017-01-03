from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import Programming.configuration as conf
from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator
from Programming.Learning.PCA_SVM import configuration_default


def plot_gallery(images, titles, h, w, depth, n_row=5, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w, depth)))
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


class PCA_SVM(object):
    def __init__(self, data_sets):
        self.init_name()
        self.time_logger = TimeCalculator(self.name)
        self.init_configuration()
        self.init_data(data_sets)

    def init_name(self):
        self.name = "PCA & SVM"

    def init_configuration(self):
        self.conf_s = configuration_default

    def init_data(self, data_sets):
        data_sets.flatten()
        self.training_data = data_sets.train.images
        self.validation_data = data_sets.validation.images if not self.conf_s.USE_TEST_DATA else data_sets.test.images

        self.train_labels = data_sets.train.labels
        self.train_labels = self.train_labels.reshape(self.train_labels.shape[0])
        self.validation_labels = data_sets.validation.labels if not self.conf_s.USE_TEST_DATA else data_sets.test.labels
        self.validation_labels = self.validation_labels.reshape(self.validation_labels.shape[0])

    def run(self):
        self.time_logger.show("Start learning")

        n_components = self.conf_s.NUM_OF_COMPONENTS
        pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True)
        pca.fit(self.training_data)
        self.time_logger.show("PCA finished")

        #eigenfaces = pca.components_.reshape((n_components, h, w, depth))
        #eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        #plot_gallery(eigenfaces, eigenface_titles, h, w, depth)
        #plt.show()

        self.training_data = pca.transform(self.training_data)
        self.validation_data = pca.transform(self.validation_data)
        self.time_logger.show("Finished transforming data sets")

        clf = GridSearchCV(SVC(class_weight='balanced', random_state=conf.SEED), self.conf_s.PARAM_GRID)
        clf = clf.fit(self.training_data, self.train_labels)
        self.time_logger.show("Finished grid search")
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        y_pred = clf.predict(self.validation_data)
        y_test = self.validation_labels

        confusion_ma = confusion_matrix(y_test, y_pred, labels=range(len(conf.DATA_TYPES_USED)))
        print(classification_report(y_test, y_pred, target_names=conf.DATA_TYPES_USED))
        print(confusion_matrix(y_test, y_pred, labels=range(len(conf.DATA_TYPES_USED))))

        test_error, _, _, _ = precision_recall_fscore_support(y_test, y_pred,
                                                      average='macro')
        test_error *= 100
        test_error = 100 - test_error

        helper.write_eval_stats(confusion_ma, test_error, self.conf_s.USE_TEST_DATA)
        self.time_logger.show("Finished validation prediction")
        return test_error, confusion_ma

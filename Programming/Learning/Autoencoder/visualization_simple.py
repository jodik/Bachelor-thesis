from keras.engine import Input
from keras.engine import Model
from keras.models import load_model
from Programming.Learning.Autoencoder.autoencoder_simple import SimpleAutoEncoder
import matplotlib.pyplot as plt


def plot_gallery(images, n_row=3, n_col=6):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i])
        plt.xticks(())
        plt.yticks(())


class VisualisationSimpleAutoEncoder(SimpleAutoEncoder):

    def init_name(self):
        self.name = "VisualizeSimpleAutoEncoder"

    def run(self):
        autoencoder = load_model('Learning/Autoencoder/models/simple42.h5')

        decoded_imgs = autoencoder.predict(self.validation_data)

        #encoded_imgs = encoder.predict(self.validation_data)
        #decoded_imgs = decoder.predict(encoded_imgs)
        decoded_imgs = decoded_imgs.reshape((len(decoded_imgs), 32, 32, 3))
        original_imgs = self.validation_data.reshape(len(self.validation_data), 32, 32, 3)

        plot_gallery(decoded_imgs[:18])
        plt.show()
        plot_gallery(original_imgs[:18])
        plt.show()

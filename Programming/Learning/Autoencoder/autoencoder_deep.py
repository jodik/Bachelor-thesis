from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense
from Programming.Learning.Autoencoder import configuration_deep
from Programming.Learning.Autoencoder.autoencoder_simple import SimpleAutoEncoder


class DeepAutoEncoder(SimpleAutoEncoder):

    def init_name(self):
        self.name = "DeepAutoEncoder"

    def init_configuration(self):
        self.conf_s = configuration_deep

    def model(self):
        # this is our input placeholder
        input_img = Input(shape=(32*32*3,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(32*32*3, activation='sigmoid')(decoded)

        return Model(input=input_img, output=decoded)

    def save_model(self, model):
        model.save('Learning/Autoencoder/models/deep.h5')

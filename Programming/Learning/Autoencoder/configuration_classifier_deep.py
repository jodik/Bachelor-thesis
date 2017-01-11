import keras

AUTOENCODER_MODEL = 'Learning/Autoencoder/models/deep4.h5'
NUMBER_OF_EPOCHS = 750
NUMBER_OF_NEURONS_IN_HIDDEN_LAYER = 170
OPTIMIZER = keras.optimizers.SGD(lr=0.01, momentum=0, decay=0, nesterov=False)
USE_TEST_DATA = True
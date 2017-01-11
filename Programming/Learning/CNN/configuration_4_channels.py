import Programming.configuration as conf

FC1_FEATURES = 1350
BASE_LEARNING_RATE = 0.0005
DECAY_RATE = 0.6
MOMENTUM = 0.95
DROPOUT_PROBABILITY = 0.3
DECAY_STEP_X_TIMES_TRAIN_SIZE = 14
CON_FIRST_STRIDE = 2
CONV_FIRST_FILTER_SIZE = 5
CONV_SECOND_FILTER_SIZE = 5
CONV_FIRST_DEPTH = 60
POOL_FIRST_SIZE = 2
CONV_SECOND_DEPTH = 130
POOL_SEC_SIZE = 2
EVAL_FREQUENCY = 10
VALIDATION_FREQUENCY = 30
TRAIN_VALIDATION_CONDITION = 30
BATCH_SIZE = 100
NUM_CHANNELS = 4
USE_TEST_DATA = False
SCALE = 2

IMAGE_HEIGHT = 16 * SCALE
IMAGE_WIDTH = 16 * SCALE

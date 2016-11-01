# By user
SCALE = 2
BATCH_SIZE = 100
TRAIN_VALIDATION_CONDINATION = 15
EVAL_BATCH_SIZE = 151
EVAL_FREQUENCY = 30  # Number of steps between evaluations.
SEED = 66478  # Set to None for random seed.
TEST_PERCENTAGE = 20
VALIDATION_PERCENTAGE = 20
DATA_TYPES_USED = ['Blue','Green', 'White', 'Box', 'Can', 'Chemical', 'Colorful']
BASE_LEARNING_RATE = 0.0005
DECAY_RATE = 0.9
HARD_DIFFICULTY = True
PERMUTATION_INDEX = 3
FULL_CROSS_VALIDATION = False
CROPPED_VERSION = True
MOMENTUM = 0.95
DROPOUT_PROBABILITY = 0.6
CROSS_VALIDATION_ITERATIONS = 5
EXTENDED_DATASET = True
DECAY_STEP_X_TIMES_TRAIN_SIZE = 8

# In case two convolutional layers
FC1_FEATURES = 1300
CON_FIRST_STRIDE = 2
CONV_FIRST_DEPTH = 75
POOL_FIRST_SIZE = 2
CONV_SECOND_DEPTH = 150
POOL_SEC_SIZE = 2

# Should not change
DATA_TYPES_USED = sorted(DATA_TYPES_USED)
NUM_LABELS = len(DATA_TYPES_USED)
NUM_CHANNELS = 3
PIXEL_DEPTH = 255.0
IMAGE_WIDTH = 16 * SCALE
if CROPPED_VERSION:
    IMAGE_HEIGHT = 16 * SCALE
    SOURCE_FOLDER_NAME = "../../../Datasets/Cropped datasets/Dataset_"+str(IMAGE_WIDTH)+"_"+str(IMAGE_HEIGHT)+"/";
else:
    IMAGE_HEIGHT = 12 * SCALE
    SOURCE_FOLDER_NAME = "../../../Datasets/Original datasets/Dataset_"+str(IMAGE_WIDTH)+"_"+str(IMAGE_HEIGHT)+"/";
PERMUTATION_FOLDER_NAME = "../../../Programming/Permutations/";
ALL_DATA_TYPES = ['Blue',
 'Box',
 'Can',
 'Chemical',
 'Colorful',
 'Green',
 'Multiple Objects',
 'Nothing',
 'White']
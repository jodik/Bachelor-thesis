# By user
SCALE = 2
BATCH_SIZE = 450
NUM_EPOCHS = 1000
EVAL_BATCH_SIZE = 151
EVAL_FREQUENCY = 30  # Number of steps between evaluations.
SEED = 66478  # Set to None for random seed.
TEST_PERCENTAGE = 20
VALIDATION_PERCENTAGE = 20
DATA_TYPES_USED = ['Blue','Green', 'White', 'Box', 'Can', 'Chemical', 'Colorful', 'Nothing']
BASE_LEARNING_RATE = 0.008
DECAY_RATE = 0.99
HARD_DIFFICULTY = False
PERMUTATION_INDEX = 3

# Should not change
DATA_TYPES_USED = sorted(DATA_TYPES_USED)
NUM_LABELS = len(DATA_TYPES_USED)
IMAGE_WIDTH = 16 * SCALE
IMAGE_HEIGHT = 12 * SCALE
NUM_CHANNELS = 3
PIXEL_DEPTH = 255.0
SOURCE_FOLDER_NAME = "../../../Datasets/Dataset_"+str(IMAGE_WIDTH)+"_"+str(IMAGE_HEIGHT)+"/";
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
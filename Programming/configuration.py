# By user
EXTENDED_DATASET = False
BLACK_BORDER = True
FULL_CROSS_VALIDATION = False
HARD_DIFFICULTY = True
CROPPED_VERSION = True
PERMUTATION_INDEX = 3
NUM_CHANNELS_PIC = 3
NUM_CHANNELS_EDGES = 1
SEED = 66478  # Set to None for random seed.
TEST_PERCENTAGE = 20
VALIDATION_PERCENTAGE = 20
DATA_TYPES_USED = sorted(['Blue', 'Box', 'Can', 'Chemical', 'Colorful', 'Green', 'White'])


# Should not change
NUM_LABELS = len(DATA_TYPES_USED)

PIXEL_DEPTH = 255.0

SUB_FOLDER = 'Bordered with black color/' if BLACK_BORDER else 'Extended with itself/'

ALL_DATA_TYPES = sorted(['Blue',
 'Box',
 'Can',
 'Chemical',
 'Colorful',
 'Green',
 'Multiple Objects',
 'Nothing',
 'White'])
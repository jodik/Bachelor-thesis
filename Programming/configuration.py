EXTENDED_DATASET = True
BLACK_BORDER = True
HARD_DIFFICULTY = True
SIMPLIFIED_CATEGORIES = False
CROPPED_VERSION = True
WRITE_TO_FILE = True
FULL_CROSS_VALIDATION = True
CROSS_VALIDATION_ITERATIONS = 5
SEED = 66478
TEST_PERCENTAGE = 20
PERMUTATION_INDEX = 3

NUM_CHANNELS_PIC = 3
NUM_CHANNELS_EDGES = 1
DATA_TYPES_USED = sorted(['Blue', 'Box', 'Can', 'Chemical', 'Colorful', 'Green', 'White'])


# Should not change
NUM_LABELS = 3 if SIMPLIFIED_CATEGORIES else len(DATA_TYPES_USED)

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


def update_simplified_categories(self, val):
    self.SIMPLIFIED_CATEGORIES = val
    self.NUM_LABELS = 3 if self.SIMPLIFIED_CATEGORIES else len(self.DATA_TYPES_USED)

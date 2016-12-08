import Programming.Learning.CNN.configuration_edges as confs
# By user
SCALE = 2
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
IMAGE_WIDTH = 16 * SCALE

SUB_FOLDER = 'Bordered with black color/' if BLACK_BORDER else 'Extended with itself/'
if CROPPED_VERSION:
    IMAGE_HEIGHT = 16 * SCALE
    SOURCE_FOLDER_NAME = "../Datasets/Cropped datasets/"+SUB_FOLDER+"Dataset_"+str(IMAGE_WIDTH)+"_"+str(IMAGE_HEIGHT)+"/";
    SOURCE_FOLDER_NAME2 = "../Datasets/Cropped datasets/" + SUB_FOLDER + "Dataset_"+str(confs.IMAGE_WIDTH)+"_"+str(confs.IMAGE_HEIGHT)+"/";
else:
    IMAGE_HEIGHT = 12 * SCALE
    SOURCE_FOLDER_NAME = "../Datasets/Original datasets/Dataset_"+str(IMAGE_WIDTH)+"_"+str(IMAGE_HEIGHT)+"/";

ALL_DATA_TYPES = sorted(['Blue',
 'Box',
 'Can',
 'Chemical',
 'Colorful',
 'Green',
 'Multiple Objects',
 'Nothing',
 'White'])
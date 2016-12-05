from cobs import cobs
import os
import cv2

TYPE_DIRECTORY = 'Cropped'  # or Original
BLACK_BORDER = True
SUB_FOLDER = 'Bordered with black color/' if BLACK_BORDER else 'Extended with itself/'
IMAGES_FOLDER_NAME = 'Images/' + TYPE_DIRECTORY + ' images/' + SUB_FOLDER
DATASETS_FOLDER = 'Datasets/' + TYPE_DIRECTORY + ' datasets/' + SUB_FOLDER

SCALE = 3
WIDTH = 16 * SCALE
HEIGHT = 16 * SCALE
DATASET_NAME = "Dataset_" + str(WIDTH) + "_" + str(HEIGHT) + '/'
DATASET_FOLDER = DATASETS_FOLDER + DATASET_NAME


def retrieve_folder(file_path):
    file_name = file_path.split('/')
    lenght_file_name = len(file_name[-1])
    return file_path[:-lenght_file_name]


def write_to_file(path, bytes_content):
    folder = retrieve_folder(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "wb") as compdata:
        bytes_content = bytearray(cobs.encode(bytes_content))
        compdata.write(bytes_content)
        compdata.close()


def list_images():
    f = []
    res = []
    for (dirpath, dirnames, filenames) in os.walk(IMAGES_FOLDER_NAME + DATASET_NAME):
        if len(dirnames) == 0:
            f.append((dirpath, filenames))

    for (dir_path, file_names) in f:
        for file_name in file_names:
            if file_name != '.DS_Store' and file_name != 'data.byte':
                img = cv2.imread(dir_path + '/' + file_name, cv2.IMREAD_COLOR)
                res.append((img, file_name, dir_path))

    return res

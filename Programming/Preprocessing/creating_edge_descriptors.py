from Programming.Preprocessing import general
import cv2
import numpy as np

def extendListEightTimes(l):
    l.extend(l)
    l.extend(l)
    l.extend(l)
    return l

images = general.list_images()
edge_desscriptors = np.zeros((len(images), general.WIDTH, general.HEIGHT), dtype=np.uint8)
i = 0
for (img, filename, dir_path) in images:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 255, L2gradient=True)
    edge_desscriptors[i] = np.asarray(edges, dtype=np.uint8)
    i += 1

edge_desscriptors = edge_desscriptors.reshape(-1)
general.write_to_file(general.DATASET_FOLDER + 'edges.byte', edge_desscriptors)


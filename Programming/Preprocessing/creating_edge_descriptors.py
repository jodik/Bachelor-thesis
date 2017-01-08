from Programming.Preprocessing import general
import cv2
import numpy as np
import matplotlib.pyplot as plt


def extendListEightTimes(l):
    l.extend(l)
    l.extend(l)
    l.extend(l)
    return l


def plot_gallery(images, h, w, depth, n_row=3, n_col=6):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i])
        #plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


images = general.list_images()
edge_desscriptors = np.zeros((len(images), general.WIDTH, general.HEIGHT), dtype=np.uint8)
i = 0
imddd = []
for (img, filename, dir_path) in images:
    imddd.append(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 255, L2gradient=True)
    edge_desscriptors[i] = np.asarray(edges, dtype=np.uint8)
    i += 1

start = 500
plot_gallery(edge_desscriptors[start:start+20], general.HEIGHT, general.WIDTH, 1)
plot_gallery(imddd[start:start+20], general.HEIGHT, general.WIDTH, 3)
plt.show()

edge_desscriptors = edge_desscriptors.reshape(-1)
#general.write_to_file(general.DATASET_FOLDER + 'edges.byte', edge_desscriptors)


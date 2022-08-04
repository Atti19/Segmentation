import numpy as np
import torch
from skimage.io import imshow
import matplotlib.pyplot as plt

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def color_map_viz():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    nclasses = 20
    row_size = 50
    col_size = 500
    cmap = color_map()
    array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
    array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]

    imshow(array)
    plt.yticks([row_size*i+row_size/2 for i in range(nclasses+1)], labels)
    plt.xticks([])
    plt.show()

culori=[
     [0, 0, 0],
     [102, 0, 0],
     [102, 102, 0],
     [51, 102, 0],
     [0, 102, 102],
     [0, 0, 102],
     [51, 0, 102],
     [102, 0, 102],
     [102, 0, 51],
     [255, 255, 51],
     [153, 255, 51],
     [51, 255, 255],
     [51, 153, 255],
     [178, 102, 255],
     [160, 160, 160],
     [255, 153, 153],
     [255, 104, 153],
     [153, 255, 255],
     [255, 153, 255],
     [255, 255, 255],
     [204, 204, 255],
]
def toColor(array):
    colored_array = np.empty([3, 256, 256])
    array = torch.squeeze(array)
    for i in range(256):
        for j in range(256):
            index = array[i][j]
            colored_array[0][i][j] = culori[index][0]
            colored_array[1][i][j] = culori[index][1]
            colored_array[2][i][j] = culori[index][2]
    return colored_array
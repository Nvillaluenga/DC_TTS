import numpy as np
import os
from matplotlib import pyplot as plt


def plot(arr, title='Default'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    im = ax.imshow(np.rot90(arr), interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax)
    plt.show()


def visualizeMag(fileName):
    img_array = np.load(os.path.join("mags", fileName))
    plot(img_array, f"Mag: {fileName}")


def visualizeMel(fileName):
    img_array = np.load(os.path.join("mels", fileName))
    plot(img_array, f"Mel: {fileName}")


if __name__ == "__main__":
    visualizeMag("LJ001-0001.npy")
    visualizeMel("LJ001-0001.npy")

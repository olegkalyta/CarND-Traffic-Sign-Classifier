import cv2
import numpy as np

def load_images():
    images = np.zeros((5,32,32,3), dtype=np.float)
    for i in range(0, 5):
        images[i] = cv2.imread('data/images/{}.jpg'.format(i + 1))

    return images

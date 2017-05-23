import cv2
import numpy as np

def normalizeImage(image):
    image = image.astype('float')
    for rowIndex in list(np.arange(32)):
        for colIndex in list(np.arange(32)):
            for i in range(3):
                image[rowIndex][colIndex][i] = (image[rowIndex][colIndex][i] - 128) / 128
    return image

def load_images():
    images = np.zeros((5,32,32,3), dtype=np.float)
    for i in range(0, 5):
        temp = cv2.imread('test_images/{}.jpg'.format(i + 1))
        images[i] = normalizeImage(temp)


    return images

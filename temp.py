import numpy as np
import cv2
import os

def process(path):
    image = cv2.imread(path)
    # print(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpg', image)
    image = cv2.resize(image, (512,512), interpolation =cv2.INTER_AREA)
    cv2.imwrite('resize_image.jpg', image)
    # _, image = cv2.threshold(image,190,255,cv2.THRESH_BINARY)
    # cv2.imwrite('small_process.jpg', image)
    # print(np.std(image))
    # return np.float32(image - np.mean(image)) / np.std(image)
    return np.float32(image/255)


process(os.path.join('sample','1_10.tif'))
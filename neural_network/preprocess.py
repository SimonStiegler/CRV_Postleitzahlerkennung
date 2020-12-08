import cv2
import numpy as np

def preprocess_segmented_image(img):
    img = ~img
    s = max(img.shape[0:2])
    f = np.zeros((s, s, 3), np.uint8)
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2
    f[ay:img.shapep[0]+ay, ax:ax+img.shape[1]] = img
    img = cv2.resize(img, (28,28))
    return img
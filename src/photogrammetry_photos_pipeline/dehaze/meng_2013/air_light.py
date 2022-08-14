import cv2
import numpy as np


def air_light(HazeImg, AirlightMethod, windowSize):
    if AirlightMethod.lower() == 'fast':
        a = []
        if (len(HazeImg.shape) == 3):
            for ch in range(len(HazeImg.shape)):
                kernel = np.ones((windowSize, windowSize), np.uint8)
                min_img = cv2.erode(HazeImg[:, :, ch], kernel)
                a.append(int(min_img.max()))
        else:
            kernel = np.ones((windowSize, windowSize), np.uint8)
            min_img = cv2.erode(HazeImg, kernel)
            a.append(int(min_img.max()))
    return a

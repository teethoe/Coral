import cv2
import numpy as np

class skeleton:
    def __init__(self, mask):
        self.mask = mask

    def skeletize(self, element):
        size = np.size(self.mask)
        skel = np.zeros(self.mask.shape, np.uint8)
        fin = self.mask.copy()
        done = False
        while not done:
            eroded = cv2.erode(fin, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(fin, temp)
            skel = cv2.bitwise_or(skel, temp)
            fin = eroded.copy()

            zeros = size - cv2.countNonZero(fin)
            if zeros == size:
                done = True

        return skel
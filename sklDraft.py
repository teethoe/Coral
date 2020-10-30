import cv2
import numpy as np
from process import Process

imga = cv2.imread('./img/Coral after.png')
aft = Process(imga)

kernel = np.ones((5,5),np.uint8)
aft.lowp = np.array([150,60,100])
aft.highp = np.array([170,255,255])
aft.loww = np.array([90,15,150])
aft.highw = np.array([115,255,255])

maska = aft.mask(kernel)

size = np.size(maska)
skel = np.zeros(maska.shape, np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
fin = maska.copy()

while not done:
    eroded = cv2.erode(fin, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(fin, temp)
    skel = cv2.bitwise_or(skel, temp)
    fin = eroded.copy()

    zeros = size - cv2.countNonZero(fin)
    if zeros == size:
        done = True

cv2.imshow('skel', skel)
cv2.waitKey(0)
cv2.destroyAllWindows()


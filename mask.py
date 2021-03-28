import cv2
import numpy as np

img = cv2.imread('./img/Coral before.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernel = np.ones((5,5),np.uint8)
lowp = np.array([150,60,100])
highp = np.array([170,255,255])
loww = np.array([90,15,150])
highw = np.array([115,255,255])
maskp = cv2.inRange(hsv, lowp, highp)
maskw = cv2.inRange(hsv, loww, highw)
mask = maskp + maskw
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
res = cv2.bitwise_and(img, img, mask=opening)
cv2.imshow('Before', img)
cv2.imshow('Mask', mask)
cv2.imshow('Res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from process import Process

kernel = np.ones((5,5),np.uint8)

img = cv2.imread('./img/Coral before.png')
bef = Process(img)

bef.lowp = np.array([150,60,100])
bef.highp = np.array([170,255,255])
bef.loww = np.array([90,15,150])
bef.highw = np.array([115,255,255])

mask = bef.mask(kernel)
cv2.imshow('mask', mask)
res = bef.res(mask)

blur = cv2.GaussianBlur(mask,(5,5),0)
cv2.imshow('blur', blur)
edges = cv2.Canny(blur,100,200,apertureSize = 3)
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
erosion = cv2.erode(gradient,kernel,iterations = 1)
cv2.imshow('edges', edges)
#cv2.imshow('gradient', gradient)
#cv2.imshow('erosion', erosion)

minLineLength = 20
maxLineGap = 10
linesEd = cv2.HoughLinesP(edges,1,90*np.pi/180,5,minLineLength,maxLineGap)
linesGr = cv2.HoughLinesP(gradient,1,np.pi/180,5,minLineLength,maxLineGap)
linesEr = cv2.HoughLinesP(erosion,1,np.pi/180,5,minLineLength,maxLineGap)


def add_lines(lines, img):
    final = img.copy()
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(final,(x1,y1),(x2,y2),(0,255,0),2)
    return final


edgeFin = add_lines(linesEd, img)
gradFin = add_lines(linesGr, img)
erodFin = add_lines(linesEr, img)

gray = np.float32(mask)
dst = cv2.cornerHarris(gray,	2,	3,	0.04)
img[dst>0.01 *	dst.max()]   =	[0,	0,	255]

cv2.imshow('edgesF', edgeFin)
cv2.imshow('corners', img)
#cv2.imshow('gradientF', gradFin)
#cv2.imshow('erosionF', erodFin)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
from process import Process, fix
from change import Change, plot
from skeleton import Skeleton

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)

imgb = cv2.imread('./img/ref/Coral before.png')
bef = Process(imgb)

bef.lowp = np.array([150, 60, 100])
bef.highp = np.array([170, 255, 255])
bef.loww = np.array([90, 15, 150])
bef.highw = np.array([115, 255, 255])

maskb, imgb = bef.mask(kernel)
sbef = Skeleton(maskb)
skelb = sbef.skeletize()

imga = cv2.imread('./img/ref/Coral after.png')
aft = Process(imga)

aft.lowp = np.array([150, 60, 100])
aft.highp = np.array([170, 255, 255])
aft.loww = np.array([90, 15, 150])
aft.highw = np.array([115, 255, 255])

maska, imga = aft.mask(kernel)
saft = Skeleton(maska)
skela = saft.skeletize()

skelb = fix(skelb, skela)
skelb = np.uint8(np.where(skelb > 0, 255, 0))
skela = np.uint8(np.where(skela > 0, 255, 0))
skelb = cv2.morphologyEx(skelb, cv2.MORPH_DILATE, kernel)
skela = cv2.morphologyEx(skela, cv2.MORPH_DILATE, kernel)
dif = Change(imga, skelb, skela)
growth, death = dif.growth_death(kernel2)
final = dif.final()

cv2.imshow('skelb', skelb)
cv2.imshow('skela', skela)
plot(imgb, imga, final)
cv2.waitKey(0)
cv2.destroyAllWindows()


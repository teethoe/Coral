import cv2
import numpy as np
import matplotlib.pyplot as plt
from process import Process, fix
from change import Change, plot


kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)

imgb = cv2.imread('./img/before.png')
imga = cv2.imread('./img/3.jpg')
imgb = cv2.resize(imgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
imga = cv2.resize(imga, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)

pbef = Process(imgb)
pbef.lowp = np.array([145, 80, 120])
pbef.highp = np.array([175, 255, 255])
pbef.loww = np.array([85, 20, 160])
pbef.highw = np.array([115, 100, 255])
maskb, imgb = pbef.mask(kernel)
resb = pbef.res(maskb)


paft = Process(imga)
paft.cluster(10)
paft.pie(10)
cv2.waitKey(0)
paft.loww = np.array([90, 0, 150])
paft.highw = np.array([120, 70, 255])
maska, imga = paft.mask(kernel)
resa = paft.res(maska)

cv2.imshow('maskb', maskb)
cv2.imshow('maska', maska)
cv2.waitKey(0)

maskb = fix(maskb, maska)
maskb = np.uint8(np.where(maskb > 0, 255, 0))
maska = np.uint8(np.where(maska > 0, 255, 0))
resb = fix(resb, resa)
whiteb = pbef.bleached(resb, kernel2)
whitea = paft.bleached(resa, kernel2)
dif = Change(imga, maskb, maska)
growth, death = dif.growth_death(kernel2)
bleach, recover = dif.bleach_recover(whiteb, whitea, growth, death, kernel)
final = dif.final()

plot(imgb, imga, final)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

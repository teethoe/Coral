import cv2
import numpy as np
import matplotlib.pyplot as plt
from process import Process, fix
from change import Change, plot


kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)

imgb = cv2.imread('./img/ref/new/2.jpg')
#imgb = cv2.flip(imgb, 1)
imga = cv2.imread('./img/ref/new/F.png')


imgb = cv2.resize(imgb, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
pbef = Process(imgb)
pbef.lowp = np.array([145, 0, 0])
pbef.highp = np.array([175, 255, 255])
pbef.loww = np.array([90, 0, 150])
pbef.highw = np.array([120, 50, 255])
maskb, imgb = pbef.mask(kernel)
resb = pbef.res(maskb)
cv2.imshow('maskb', maskb)
cv2.waitKey(0)


imga = cv2.resize(imga, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
paft = Process(imga)
paft.lowp = np.array([145, 0, 0])
paft.highp = np.array([175, 255, 255])
paft.loww = np.array([90, 0, 150])
paft.highw = np.array([120, 50, 255])
maska, imga = paft.mask(kernel)
resa = paft.res(maska)


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

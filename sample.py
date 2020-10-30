import cv2
import numpy as np
from process import Process, fix
from skanf import skn
import matplotlib.pyplot as plt
import scipy.ndimage


kernel = np.ones((5,5),np.uint8)
#element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

imgb = cv2.imread('./img/ref/Coral before.png')
imga = cv2.imread('./img/ref/Coral after.png')


pbef = Process(imgb)
pbef.lowp = np.array([150,60,100])
pbef.highp = np.array([170,255,255])
pbef.loww = np.array([90,15,150])
pbef.highw = np.array([115,255,255])
maskb, _ = pbef.mask(kernel)
resb = pbef.res(maskb)


paft = Process(imga)
paft.lowp = np.array([150,60,100])
paft.highp = np.array([170,255,255])
paft.loww = np.array([90,15,150])
paft.highw = np.array([115,255,255])
maska, _ = paft.mask(kernel)
resa = paft.res(maska)
maskn = fix(maskb, maska)

maskf = maska - maskn
kernel2 = np.ones((8,8),np.uint8)
maskf = cv2.morphologyEx(maskf, cv2.MORPH_OPEN, kernel2)
cv2.imshow('maskf', maskf)
contours, hierarchy = cv2.findContours(maskf, 1, 2)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
imgf = cv2.rectangle(imga,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('cnt', imgf)

#sbef = skn(maskb)
#skanb = sbef.skan(1)
#skanb.to_csv(r'./before.csv')

#saft = skn(maska)
#skana = saft.skan(2)
#skana.to_csv(r'./after.csv')

#fig, ax = plt.subplots()
#plt.imshow(maska)

#plt.scatter(skanb['image-coord-src-1']*wa/wb, skanb['image-coord-src-0']*(69/80.5)+(hb*11.5/80.5), c='#00ffff')
#plt.scatter(skanb['image-coord-dst-1']*wa/wb, skanb['image-coord-src-0']*(69/80.5)+(hb*11.5/80.5), c='#ff0000')
#plt.scatter(skana['image-coord-src-1'], skana['image-coord-src-0'], c='#0000ff')
#plt.scatter(skana['image-coord-dst-1'], skana['image-coord-src-0'], c='#ffaa00')
#plt.xlim(0, hb)
#plt.ylim(0, wa)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
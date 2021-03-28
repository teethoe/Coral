import cv2
import numpy as np
from process import Process, fix
from change import Change


kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((10,10),np.uint8)

imgb = cv2.imread('./img/lib/red/a.jpg')
#imgb = cv2.flip(imgb, 1)
imga = cv2.imread('./img/lib/red/l.jpg')


imgb = cv2.resize(imgb,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_CUBIC)
pbef = Process(imgb)
pbef.lowp = np.array([175,255,255])
pbef.highp = np.array([10,0,0])
pbef.loww = np.array([0,0,200])
pbef.highw = np.array([20,50,255])
maskb, imgb = pbef.mask(kernel)
resb = pbef.res(maskb)


imga = cv2.resize(imga,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_CUBIC)
paft = Process(imga)
paft.lowp = np.array([170,255,255])
paft.highp = np.array([10,50,50])
paft.loww = np.array([0,0,150])
paft.highw = np.array([20,50,255])
maska, imga = paft.mask(kernel)
resa = paft.res(maska)


maskb = fix(maskb, maska)
resb = fix(resb, resa)
whiteb = pbef.bleached(resb, kernel2)
whitea = paft.bleached(resa, kernel2)
dif = Change(imga, maskb, maska)
growth, death = dif.growth_death(kernel2)
bleach, recover = dif.bleach_recover(whiteb, whitea, growth, death, kernel2)
final = dif.final()


cv2.imshow('maskb', maskb)
cv2.imshow('maska', maska)
cv2.imshow('whiteb', whiteb)
cv2.imshow('whitea', whitea)
#cv2.imshow('growth', growth)
#cv2.imshow('death', death)
#cv2.imshow('bleach', bleach)
#cv2.imshow('recover', recover)
#cv2.imshow('resb', resb)
cv2.imshow('imgb', imgb)
cv2.imshow('imga', imga)
cv2.imshow('final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()

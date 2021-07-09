import cv2
import numpy as np
from process2 import Process2, fix
from change import Change, plot
from trackbar import TrackbarWindow

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)

imgb = cv2.imread('./img/before.png')
imga = cv2.imread('./img/F.png')

imgb = cv2.resize(imgb, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
pbef = Process2(imgb)
imga = cv2.resize(imga, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
paft = Process2(imga)


btb = TrackbarWindow('Before', imgb)

while True:
    pbef.lowp, pbef.highp, pbef.loww, pbef.highw = btb.get_range(15, 15)

    lb, openingb = pbef.mask(kernel)
    cv2.imshow('openingb', openingb)
    cv2.imshow('imgb', imgb)
    x, y, w, h = lb

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        if y + h <= imgb.shape[0] and x + w <= imgb.shape[1]:
            imgb = imgb[y:y + h, x:x + w]
            maskb = cv2.blur(openingb[y:y + h, x:x + w], (5, 5))
            resb = pbef.res(imgb, maskb)
        break

atb = TrackbarWindow('After', imga)

while True:
    paft.lowp, paft.highp, paft.loww, paft.highw = atb.get_range(15, 8)

    la, openinga = paft.mask(kernel)
    cv2.imshow('openinga', openinga)
    cv2.imshow('imga', imga)
    x, y, w, h = la

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        if y + h <= imga.shape[0] and x + w <= imga.shape[1]:
            imga = imga[y:y + h, x:x + w]
            maska = cv2.blur(openinga[y:y + h, x:x + w], (5, 5))
            resa = paft.res(imga, maska)
        break


maskb = fix(maskb, maska)
maskb = np.uint8(np.where(maskb > 0, 255, 0))
maska = np.uint8(np.where(maska > 0, 255, 0))
resb = fix(resb, resa)
whiteb = pbef.bleached(resb, kernel2)
whitea = paft.bleached(resa, kernel2)
#maskc = cv2.bitwise_xor(maskb, maska)
#maskc = cv2.morphologyEx(maskc, cv2.MORPH_OPEN, kernel)
#maskc = cv2.morphologyEx(openingc, cv2.MORPH_CLOSE, kernel4)
dif = Change(imga, maskb, maska)
growth, death = dif.growth_death(kernel2)
bleach, recover = dif.bleach_recover(whiteb, whitea, growth, death, kernel)
final = dif.final()

'''cv2.imshow('growth', growth)
cv2.imshow('death', death)
cv2.imshow('maskb', maskb)
cv2.imshow('maska', maska)
cv2.imshow('whiteb', whiteb)
cv2.imshow('whitea', whitea)'''
#cv2.imshow('maskc', maskc)

plot(imgb, imga, final)

cv2.waitKey(0)
cv2.destroyAllWindows()

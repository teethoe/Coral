import cv2
import numpy as np
from process2 import Process2, fix
from change import Change, plot
from trackbar import TrackbarWindow

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((15, 15), np.uint8)
kernel5 = np.ones((2, 2), np.uint8)
kernel6 = np.ones((8, 8), np.uint8)


cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read(1)
    frame = cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        imga = frame
        break

cap.release()
cv2.destroyAllWindows()

print(imga.shape)
cv2.imshow('imga', imga)


imgb = cv2.imread('./img/lib/pink/before/a.jpg')
#imgb = cv2.flip(imgb, 1)
imgb = cv2.resize(imgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
imga = cv2.resize(imga, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
pbef = Process2(imgb)
paft = Process2(imga)

btb = TrackbarWindow('Before', imgb)

while True:
    #pbef.lowp, pbef.highp, pbef.loww, pbef.highw = btb.get_range(10, 10)
    pbef.lowp = np.array([157, 63, 0])
    pbef.highp = np.array([177, 255, 255])
    pbef.loww = np.array([178, 13, 0])
    pbef.highw = np.array([18, 255, 255])

    lb, openingb = pbef.mask(kernel)
    cv2.imshow('openingb', openingb)
    cv2.imshow('imgb', imgb)

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        x, y, w, h = lb
        if y + h <= imgb.shape[0] and x + w <= imgb.shape[1]:
            imgb = imgb[y:y + h, x:x + w]
            maskb = cv2.blur(openingb[y:y + h, x:x + w], (5, 5))
            resb = pbef.res(imgb, maskb)
        break


def nothing(x):
    pass


atb = TrackbarWindow('After', imga)
cv2.createTrackbar('prange', 'After', 0, 20, nothing)
cv2.createTrackbar('wrange', 'After', 0, 20, nothing)


while True:
    p = cv2.getTrackbarPos('prange', 'After') + 5
    w = cv2.getTrackbarPos('wrange', 'After') + 5
    paft.lowp, paft.highp, paft.loww, paft.highw = atb.get_range(p, w)

    thv = atb.get_threshold()
    lt, closing = paft.maskth(imga, thv, kernel5)
    cv2.imshow('thresh', closing)

    la, openinga = paft.mask(kernel6)
    cv2.imshow('openinga', openinga)
    cv2.imshow('imga', imga)

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        x, y, w, h = la
        if y + h <= imga.shape[0] and x + w <= imga.shape[1]:
            imga = imga[y:y + h, x:x + w]
            #maska = cv2.blur(openinga[y:y + h, x:x + w], (5, 5))
            maska = openinga[y:y + h, x:x + w]
            resa = paft.res(imga, maska)
        break

maska = cv2.morphologyEx(maska, cv2.MORPH_DILATE, kernel5)
cv2.imshow('maska', maska)
cv2.waitKey(0)


maskb = fix(maskb, maska)
cv2.imshow('maskb', maskb)
cv2.waitKey(0)
maskb = np.uint8(np.where(maskb > 0, 255, 0))
maska = np.uint8(np.where(maska > 0, 255, 0))
#print(np.unique(maskb))
resb = fix(resb, resa)
whiteb = pbef.bleached(resb, kernel2)
whitea = paft.bleached(resa, kernel2)
maskc = cv2.bitwise_xor(maskb, maska)
maskc = cv2.morphologyEx(maskc, cv2.MORPH_OPEN, kernel)
#maskc = cv2.morphologyEx(openingc, cv2.MORPH_CLOSE, kernel4)
dif = Change(imga, maskb, maska)
growth, death = dif.growth_death(kernel2)
bleach, recover = dif.bleach_recover(whiteb, whitea, growth, death, kernel2)
final = dif.final()


cv2.imshow('maskb', maskb)
cv2.imshow('maska', maska)

plot(imgb, imga, final)

cv2.waitKey(0)
cv2.destroyAllWindows()

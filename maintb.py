import cv2
import numpy as np
import matplotlib. pyplot as plt
from process2 import Process2, fix
from change import Change

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)

#imgb = cv2.imread('./img/lib/red/a.jpg')
imgb = cv2.imread('./img/lib/ps/uwN ig fil.jpeg')
imga = cv2.imread('./img/lib/ps/ps1a.jpg')

#imgb = cv2.resize(imgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
imgb = cv2.flip(imgb, 1)
imgb = cv2.resize(imgb, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
pbef = Process2(imgb)
#imga = cv2.resize(imga, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
#imga = cv2.resize(imga, None, fx=1.5, fy=1, interpolation=cv2.INTER_CUBIC)
paft = Process2(imga)


def nothing():
    pass


cv2.namedWindow('Before')
cv2.imshow('Before', imgb)

cv2.createTrackbar('hpink', 'Before', 0, 180, nothing)
cv2.createTrackbar('slpink', 'Before', 0, 255, nothing)
cv2.createTrackbar('shpink', 'Before', 0, 255, nothing)
cv2.createTrackbar('vlpink', 'Before', 0, 255, nothing)
cv2.createTrackbar('vhpink', 'Before', 0, 255, nothing)

cv2.createTrackbar('hwhite', 'Before', 0, 180, nothing)
cv2.createTrackbar('slwhite', 'Before', 0, 255, nothing)
cv2.createTrackbar('shwhite', 'Before', 0, 255, nothing)
cv2.createTrackbar('vlwhite', 'Before', 0, 255, nothing)
cv2.createTrackbar('vhwhite', 'Before', 0, 255, nothing)


while True:
    hpink = cv2.getTrackbarPos('hpink', 'Before')
    slpink = cv2.getTrackbarPos('slpink', 'Before')
    shpink = cv2.getTrackbarPos('shpink', 'Before')
    vlpink = cv2.getTrackbarPos('vlpink', 'Before')
    vhpink = cv2.getTrackbarPos('vhpink', 'Before')

    hwhite = cv2.getTrackbarPos('hwhite', 'Before')
    slwhite = cv2.getTrackbarPos('slwhite', 'Before')
    shwhite = cv2.getTrackbarPos('shwhite', 'Before')
    vlwhite = cv2.getTrackbarPos('vlwhite', 'Before')
    vhwhite = cv2.getTrackbarPos('vhwhite', 'Before')

    pbef.lowp = np.array([(180 + (hpink - 10)) % 180, slpink, vlpink])
    pbef.highp = np.array([(hpink + 10) % 180, shpink, vhpink])
    pbef.loww = np.array([(180 + (hwhite - 10)) % 180, slwhite, vlwhite])
    pbef.highw = np.array([(hwhite + 10) % 180, shwhite, vhwhite])
    #pbef.loww = np.array([0, 0, 150])
    #pbef.highw = np.array([180, 50, 255])

    lb, openingb = pbef.mask(kernel)
    cv2.imshow('openingb', openingb)
    cv2.imshow('imgb', imgb)
    x, y, w, h = lb

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        if y + h <= imgb.shape[0] and x + +w <= imgb.shape[1]:
            imgb = imgb[y:y + h, x:x + w]
            maskb = cv2.blur(openingb[y:y + h, x:x + w], (5, 5))
            resb = pbef.res(imgb, maskb)
        break

cv2.namedWindow('After')
cv2.imshow('After', imga)

cv2.createTrackbar('hpink', 'After', 0, 180, nothing)
cv2.createTrackbar('slpink', 'After', 0, 255, nothing)
cv2.createTrackbar('shpink', 'After', 0, 255, nothing)
cv2.createTrackbar('vlpink', 'After', 0, 255, nothing)
cv2.createTrackbar('vhpink', 'After', 0, 255, nothing)

cv2.createTrackbar('hwhite', 'After', 0, 180, nothing)
cv2.createTrackbar('slwhite', 'After', 0, 255, nothing)
cv2.createTrackbar('shwhite', 'After', 0, 255, nothing)
cv2.createTrackbar('vlwhite', 'After', 0, 255, nothing)
cv2.createTrackbar('vhwhite', 'After', 0, 255, nothing)

while True:
    hpink = cv2.getTrackbarPos('hpink', 'After')
    slpink = cv2.getTrackbarPos('slpink', 'After')
    shpink = cv2.getTrackbarPos('shpink', 'After')
    vlpink = cv2.getTrackbarPos('vlpink', 'After')
    vhpink = cv2.getTrackbarPos('vhpink', 'After')

    hwhite = cv2.getTrackbarPos('hwhite', 'After')
    slwhite = cv2.getTrackbarPos('slwhite', 'After')
    shwhite = cv2.getTrackbarPos('shwhite', 'After')
    vlwhite = cv2.getTrackbarPos('vlwhite', 'After')
    vhwhite = cv2.getTrackbarPos('vhwhite', 'After')

    paft.lowp = np.array([(180 + (hpink - 10)) % 180, slpink, vlpink])
    paft.highp = np.array([(hpink + 10) % 180, shpink, vhpink])
    paft.loww = np.array([(180 + (hwhite - 2)) % 180, slwhite, vlwhite])
    paft.highw = np.array([(hwhite + 2) % 180, shwhite, vhwhite])
    #paft.loww = np.array([0, 0, 150])
    #paft.highw = np.array([180, 50, 255])

    la, openinga = paft.mask(kernel)
    cv2.imshow('openinga', openinga)
    cv2.imshow('imga', imga)
    x, y, w, h = la

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        if y + h <= imga.shape[0] and x + +w <= imga.shape[1]:
            imga = imga[y:y + h, x:x + w]
            maska = cv2.blur(openinga[y:y + h, x:x + w], (5, 5))
            resa = paft.res(imga, maska)
        break

cntaft = maska.copy()
cntmask = maska.copy()
cntshow = maska.copy()
cntshow = cv2.cvtColor(cntshow, cv2.COLOR_GRAY2BGR)
contours, hierarchy = cv2.findContours(maska, 1, 2)
maxa = -1
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area > maxa:
        index = i
        maxa = area
for i in range(len(contours)):
    if i != index:
        cntshow = cv2.drawContours(cntshow, contours, i, (255,0,0), 2)
        cntmask = cv2.drawContours(cntmask, contours, i, 0, -1)
maska = cv2.bitwise_and(maska, maska, mask=cntmask)
resa = paft.res(imga, maska)
cv2.imshow('cntshow', cntshow)

maskb = fix(maskb, maska)
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
#cv2.imshow('whiteb', whiteb)
#cv2.imshow('whitea', whitea)
#cv2.imshow('maskc', maskc)
cv2.imshow('imgb', imgb)
cv2.imshow('imga', imga)
cv2.imshow('final', final)

fig, axs = plt.subplots(1, 3, dpi=400)
rgb_imgb = imgb[..., ::-1]
axs[0].imshow(rgb_imgb)
axs[0].set_title('Before')
axs[0].axis('off')
rgb_imga = imga[..., ::-1]
axs[1].imshow(rgb_imga)
axs[1].set_title('After')
axs[1].axis('off')
rgb_final = final[..., ::-1]
axs[2].imshow(rgb_final)
axs[2].set_title('Final')
axs[2].axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
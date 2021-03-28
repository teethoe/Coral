import cv2
import numpy as np
import random
import matplotlib. pyplot as plt
<<<<<<< HEAD
from process import Process, fix
from change import Change
import sys
=======
from process2 import Process2, fix
from change import Change
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)


def nothing():
    pass


<<<<<<< HEAD
imgk = cv2.imread('./img/lib/red/k.jpg')
imgl = cv2.imread('./img/lib/red/l.jpg')
imgk = cv2.resize(imgk, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
imgl = cv2.resize(imgl, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)

pk = Process(imgk)
pl = Process(imgl)

pk.lowp = np.array([170, 0, 0])
pk.highp = np.array([10, 255, 255])
pk.loww = np.array([0, 0, 150])
pk.highw = np.array([180, 50, 255])

pl.lowp = np.array([170, 0, 0])
pl.highp = np.array([10, 255, 255])
pl.loww = np.array([0, 0, 150])
pl.highw = np.array([180, 50, 255])

maskk, imgk = pk.mask(kernel)
maskl, imgl = pl.mask(kernel)

cv2.imshow('maskk', maskk)
cv2.imshow('maskl', maskl)

maskd = cv2.bitwise_xor(maskk, maskl)
cv2.imshow('maskd', maskd)

'''
while True:
    try:
        #file = chr(random.randint(97, 116))
        file = chr(random.randint(107, 108))
        imgb = cv2.imread('./img/lib/red/{}.jpg'.format(file))
        imga = cv2.imread('./img/lib/ps/ps1a.jpg')

        imgb = cv2.flip(imgb, 1)
        imgb = cv2.resize(imgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
        pbef = Process2(imgb)
        paft = Process2(imga)

        cv2.namedWindow('Before')
        cv2.imshow('Before', imgb)

=======
key = cv2.waitKey(1) & 0xff
while key != 27:
    key = cv2.waitKey(1) & 0xff

    x = chr(random.randint(97, 116))
    imgb = cv2.imread('./img/lib/red/{}.jpg'.format(x))
    imga = cv2.imread('./img/lib/ps/ps1a.jpg')

    imgb = cv2.flip(imgb, 1)
    imgb = cv2.resize(imgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    pbef = Process2(imgb)
    paft = Process2(imga)

    cv2.namedWindow('Before')
    cv2.imshow('Before', imgb)

    while True:
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
        pbef.lowp = np.array([170, 255, 255])
        pbef.highp = np.array([10, 0, 0])
        pbef.loww = np.array([0, 0, 150])
        pbef.highw = np.array([180, 50, 255])

        lb, openingb = pbef.mask(kernel)
        cv2.imshow('openingb', openingb)
        cv2.imshow('imgb', imgb)
        x, y, w, h = lb

        if y + h <= imgb.shape[0] and x + +w <= imgb.shape[1]:
            imgb = imgb[y:y + h, x:x + w]
            maskb = cv2.blur(openingb[y:y + h, x:x + w], (5, 5))
            resb = pbef.res(imgb, maskb)


        paft.lowp = np.array([159, 0, 0])
        paft.highp = np.array([179, 255, 255])
        paft.loww = np.array([87, 0, 230])
        paft.highw = np.array([107, 243, 255])

        la, openinga = paft.mask(kernel)
        cv2.imshow('openinga', openinga)
        cv2.imshow('imga', imga)
        x, y, w, h = la

        if y + h <= imga.shape[0] and x + +w <= imga.shape[1]:
            imga = imga[y:y + h, x:x + w]
            maska = cv2.blur(openinga[y:y + h, x:x + w], (5, 5))
            resa = paft.res(imga, maska)

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
        resb = fix(resb, resa)
        whiteb = pbef.bleached(resb, kernel2)
        whitea = paft.bleached(resa, kernel2)
        maskc = cv2.bitwise_xor(maskb, maska)
        maskc = cv2.morphologyEx(maskc, cv2.MORPH_OPEN, kernel)
<<<<<<< HEAD
=======
        #maskc = cv2.morphologyEx(openingc, cv2.MORPH_CLOSE, kernel4)
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
        dif = Change(imga, maskb, maska)
        growth, death = dif.growth_death(kernel2)
        bleach, recover = dif.bleach_recover(whiteb, whitea, growth, death, kernel2)
        final = dif.final()

        cv2.imshow('maskb', maskb)
        cv2.imshow('maska', maska)
        cv2.imshow('imgb', imgb)
        cv2.imshow('imga', imga)
        cv2.imshow('final', final)

        fig, axs = plt.subplots(1, 3, dpi=400)
        rgb_imgb = imgb[..., ::-1]
        axs[0].imshow(rgb_imgb)
<<<<<<< HEAD
        axs[0].set_title('Before {}'.format(file))
=======
        axs[0].set_title('Before')
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
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

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
<<<<<<< HEAD
            print('fk you bitch')
            break

    except KeyboardInterrupt:
        sys.exit(0)
'''

=======
            break

>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
cv2.waitKey(0)
cv2.destroyAllWindows()

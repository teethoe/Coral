import cv2
import numpy as np
<<<<<<< HEAD
from process2 import Process2, fix
from change import Change, plot
from trackbar import TrackbarWindow
=======
import matplotlib. pyplot as plt
from process2 import Process2, fix
from change import Change
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((15, 15), np.uint8)
<<<<<<< HEAD
kernel5 = np.ones((2, 2), np.uint8)


#cap = cv2.VideoCapture('./vid/rov/luso red cam1 full cut.mp4')
cap = cv2.VideoCapture('./vid/rov/luso dark pink 313 front cut 2.mp4')
=======

def nothing(x):
    pass

cap = cv2.VideoCapture('./vid/av1.mp4')
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        imga = frame
<<<<<<< HEAD
        #imga = frame[0:int(frame.shape[0] / 2), int(frame.shape[1] / 2):frame.shape[1]]
        #imga = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #imga = cv2.flip(imga, 1)
        #imga = imga[0:imga.shape[0], 100:imga.shape[1]-100]
=======
        imga = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #imga = cv2.flip(imga, 1)
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
        break

cap.release()
cv2.destroyAllWindows()

<<<<<<< HEAD
imgb = cv2.imread('./img/lib/pink/before/a.jpg')
#imgb = cv2.flip(imgb, 1)
imgb = cv2.resize(imgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
pbef = Process2(imgb)
#paft = Process2(imga)

cv2.resize(imga, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
hca = cv2.convertScaleAbs(imga, alpha=2.0, beta=20)
cv2.imshow('hca', hca)
paft = Process2(hca)
edge = cv2.Canny(hca, 100, 200)
edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel5)
filtered = cv2.bilateralFilter(cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY), 7, 50, 50)
edgef = cv2.Canny(filtered, 60, 120)
edgef = cv2.morphologyEx(edgef, cv2.MORPH_CLOSE, kernel5)
cv2.imshow('edge', edge)
cv2.imshow('edgef', edgef)
'''
contours, hierarchy = cv2.findContours(edgef, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=len)
contourImg = cv2.drawContours(imga, contour, -1, (0, 255, 0), 1)
cv2.imshow("Contours", contourImg)
'''

btb = TrackbarWindow('Before', imgb)

while True:
    pbef.lowp, pbef.highp, pbef.loww, pbef.highw = btb.get_range(10, 10)
    #pbef.loww = np.array([0, 0, 150])
    #pbef.highw = np.array([180, 50, 255])
=======

imgb = cv2.imread('./img/lib/unfin/o.jpg')

imgb = cv2.resize(imgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
pbef = Process2(imgb)

cv2.namedWindow('Before')
cv2.imshow('Before', imgb)

cv2.createTrackbar('hpink', 'Before', 0, 180, nothing)
cv2.createTrackbar('slpink', 'Before', 0, 255, nothing)
cv2.createTrackbar('shpink', 'Before', 0, 255, nothing)
cv2.createTrackbar('vlpink', 'Before', 0, 255, nothing)
cv2.createTrackbar('vhpink', 'Before', 0, 255, nothing)


while True:
    hpink = cv2.getTrackbarPos('hpink', 'Before')
    slpink = cv2.getTrackbarPos('slpink', 'Before')
    shpink = cv2.getTrackbarPos('shpink', 'Before')
    vlpink = cv2.getTrackbarPos('vlpink', 'Before')
    vhpink = cv2.getTrackbarPos('vhpink', 'Before')

    pbef.lowp = np.array([(180 + (hpink - 10)) % 180, slpink, vlpink])
    pbef.highp = np.array([(hpink + 10) % 180, shpink, vhpink])
    pbef.loww = np.array([0, 0, 150])
    pbef.highw = np.array([180, 50, 255])
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e

    lb, openingb = pbef.mask(kernel)
    cv2.imshow('openingb', openingb)
    cv2.imshow('imgb', imgb)
<<<<<<< HEAD

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        x, y, w, h = lb
        if y + h <= imgb.shape[0] and x + w <= imgb.shape[1]:
            imgb = imgb[y:y + h, x:x + w]
            maskb = cv2.blur(openingb[y:y + h, x:x + w], (5, 5))
            resb = pbef.res(imgb, maskb)
        break


#tha = cv2.adaptiveThreshold(hcag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#cv2.imshow('thresh', tha)

#atb = TrackbarWindow('After', imga)
atb = TrackbarWindow('After', imga, th=True)

while True:
    paft.lowp, paft.highp, paft.loww, paft.highw = atb.get_range(15, 17)
    #paft.lowp[0] = 0
    #paft.highp[0] = 180

    thv = atb.get_threshold()
    lt, closing = paft.maskth(hca, thv, kernel5)
    cv2.imshow('thresh', closing)

    '''
    paft.lowp = np.array([0, 0, 0])
    paft.highp = np.array([180, 255, vhpink])
    paft.loww = np.array([(180 + (hwhite - 20)) % 180, slwhite, vlwhite])
    paft.highw = np.array([(hwhite + 20) % 180, shwhite, vhwhite])
    '''
    #paft.loww = np.array([0, 0, 150])
    #paft.highw = np.array([180, 50, 255])

    la, openinga = paft.mask(kernel3)
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
=======
    x, y, w, h = lb

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        if y + h <= imgb.shape[0] and x + w <= imgb.shape[1]:
            imgb = imgb[y:y + h, x:x + w]
            #maskb = cv2.blur(openingb[y:y + h, x:x + w], (5, 5))
            maskb = openingb[y:y + h, x:x + w]
            resb = pbef.res(imgb, maskb)
        break

'''
gray = cv2.cvtColor(imga,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x, y, w, h = cv2.boundingRect(cnt)
imga = imga[y:y+h, x:x+w]
'''

paft = Process2(imga)
#imga = np.uint8(imga)

cv2.namedWindow('After')
cv2.imshow('After', imga)

cv2.createTrackbar('hpink', 'After', 0, 180, nothing)
cv2.createTrackbar('slpink', 'After', 0, 255, nothing)
cv2.createTrackbar('shpink', 'After', 0, 255, nothing)
cv2.createTrackbar('vlpink', 'After', 0, 255, nothing)
cv2.createTrackbar('vhpink', 'After', 0, 255, nothing)

while True:
    hpink = cv2.getTrackbarPos('hpink', 'After')
    slpink = cv2.getTrackbarPos('slpink', 'After')
    shpink = cv2.getTrackbarPos('shpink', 'After')
    vlpink = cv2.getTrackbarPos('vlpink', 'After')
    vhpink = cv2.getTrackbarPos('vhpink', 'After')

    paft.lowp = np.array([(180 + (hpink - 10)) % 180, slpink, vlpink])
    paft.highp = np.array([(hpink + 10) % 180, shpink, vhpink])
    paft.loww = np.array([0, 0, 150])
    paft.highw = np.array([180, 50, 255])

    la, openinga = paft.mask(kernel)
    cv2.imshow('openinga', openinga)
    cv2.imshow('imga', imga)
    x, y, w, h = la

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        if y + h <= imga.shape[0] and x + +w <= imga.shape[1]:
            imga = imga[y:y + h, x:x + w]
            #maska = cv2.blur(openinga[y:y + h, x:x + w], (5, 5))
            maska = openinga[y:y + h, x:x + w]
            #resa = paft.res(imga, maska)
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
        break

cv2.imshow('maska', maska)
cv2.waitKey(0)

<<<<<<< HEAD

maskb = fix(maskb, maska)
cv2.imshow('maskb', maskb)
cv2.waitKey(0)
=======
#maskb = cv2.dilate(maskb, kernel,iterations = 1)
#maska = cv2.dilate(maska, kernel,iterations = 1)

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
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
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
<<<<<<< HEAD

plot(imgb, imga, final)

cv2.waitKey(0)
cv2.destroyAllWindows()
=======
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
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e

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


#cap = cv2.VideoCapture('./vid/rov/luso red cam1 full cut.mp4')
cap = cv2.VideoCapture('./vid/rov/luso dark pink 313 front cut 2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        imga = frame
        #imga = frame[0:int(frame.shape[0] / 2), int(frame.shape[1] / 2):frame.shape[1]]
        #imga = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #imga = cv2.flip(imga, 1)
        #imga = imga[0:imga.shape[0], 100:imga.shape[1]-100]
        break

cap.release()
cv2.destroyAllWindows()

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
        break

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

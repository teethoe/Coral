import cv2
import numpy as np

imgb = cv2.imread('./img/lib/red/a.jpg')
imgb = cv2.resize(imgb,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_CUBIC)
hsv = cv2.cvtColor(imgb, cv2.COLOR_BGR2HSV)

kernel = np.ones((5,5),np.uint8)    


def nothing(x):
    pass


cv2.namedWindow('Before')

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
    print('hi')
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
    cv2.imshow('imgb', imgb)

    lowp = np.array([(180+(hpink-10))%180,slpink,vlpink])
    print(lowp)
    highp = np.array([(hpink+10)%180,shpink,vhpink])
    loww = np.array([(180+(hwhite-10))%180,slwhite,vlwhite])
    highw = np.array([(hwhite+10)%180,shwhite,vhwhite])

    if lowp[0] < highp[0]:
        maskp = cv2.inRange(hsv, lowp, highp)
    else:
        maskp = cv2.bitwise_not(cv2.inRange(hsv, highp, lowp))
        cv2.imshow('maskp', maskp)
    if loww[0] < highp[0]:
        maskw = cv2.inRange(hsv, loww, highw)
        cv2.imshow('maskw', maskw)
    else:
        maskw = cv2.bitwise_not(cv2.inRange(hsv, highw, loww))
    temp = maskp + maskw
    closing = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    cv2.imshow('closing', closing)

    contours, hierarchy = cv2.findContours(opening, 1, 2)
    l = [0 for i in range(4)]
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= l[2] and h >= l[3]:
            l = [x, y, w, h]
    x, y, w, h = l
    

    k = cv2.waitKey(1) & 0xff
    if k==ord('q'):
        if y+h<=imgb.shape[0] and x++w<=imgb.shape[1]:
            imgb = imgb[y:y + h, x:x + w]
            maskb = cv2.blur(opening[y:y + h, x:x + w],(5,5))
            cv2.imshow('maskb', maskb)
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

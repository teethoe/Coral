import cv2
<<<<<<< HEAD
import numpy as np

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((10, 10), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((15, 15), np.uint8)

def nothing(x):
    pass

cap = cv2.VideoCapture('./vid/rov/luso red cam1 full cut.mp4')
=======


cap = cv2.VideoCapture('./vid/av4.mp4')
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
<<<<<<< HEAD
        imga = frame[0:int(frame.shape[0]/2), 0:int(frame.shape[1]/2)]
        imga = imga[0:imga.shape[0], 100:imga.shape[1]-100]
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
        break

cv2.namedWindow('After')
cv2.createTrackbar('value', 'After', 0, 255, nothing)

while True:
    value = cv2.getTrackbarPos('value', 'After')

    thresh = cv2.bitwise_not(cv2.adaptiveThreshold(imga, value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 3))
    cv2.imshow('thresh', thresh)

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

cv2.imshow('thresh', thresh)
=======
        imga = frame
        break

>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
cv2.imshow('imga', imga)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2


cap = cv2.VideoCapture('./vid/av4.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        imga = frame
        break

cv2.imshow('imga', imga)
cv2.waitKey(0)
cv2.destroyAllWindows()
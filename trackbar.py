import cv2
import numpy as np


def nothing(x):
    pass


class TrackbarWindow:
    def __init__(self, name, img, **kwargs):
        self.name = name
        self.img = img

        th = kwargs.get('th', None)

        cv2.namedWindow(name)
        #cv2.imshow(name, self.img)

        cv2.createTrackbar('hpink', self.name, 0, 180, nothing)
        cv2.createTrackbar('slpink', self.name, 0, 255, nothing)
        cv2.createTrackbar('shpink', self.name, 0, 255, nothing)
        cv2.createTrackbar('vlpink', self.name, 0, 255, nothing)
        cv2.createTrackbar('vhpink', self.name, 0, 255, nothing)

        cv2.createTrackbar('hwhite', self.name, 0, 180, nothing)
        cv2.createTrackbar('slwhite', self.name, 0, 255, nothing)
        cv2.createTrackbar('shwhite', self.name, 0, 255, nothing)
        cv2.createTrackbar('vlwhite', self.name, 0, 255, nothing)
        cv2.createTrackbar('vhwhite', self.name, 0, 255, nothing)

        if th:
            cv2.createTrackbar('threshold', self.name, 0, 255, nothing)

    def get_values(self):
        hpink = cv2.getTrackbarPos('hpink', self.name)
        slpink = cv2.getTrackbarPos('slpink', self.name)
        shpink = cv2.getTrackbarPos('shpink', self.name)
        vlpink = cv2.getTrackbarPos('vlpink', self.name)
        vhpink = cv2.getTrackbarPos('vhpink', self.name)

        hwhite = cv2.getTrackbarPos('hwhite', self.name)
        slwhite = cv2.getTrackbarPos('slwhite', self.name)
        shwhite = cv2.getTrackbarPos('shwhite', self.name)
        vlwhite = cv2.getTrackbarPos('vlwhite', self.name)
        vhwhite = cv2.getTrackbarPos('vhwhite', self.name)

        arr = [hpink, slpink, shpink, vlpink, vhpink, hwhite, slwhite, shwhite, vlwhite, vhwhite]
        return arr

    def get_range(self, prange, wrange):
        hpink = cv2.getTrackbarPos('hpink', self.name)
        slpink = cv2.getTrackbarPos('slpink', self.name)
        shpink = cv2.getTrackbarPos('shpink', self.name)
        vlpink = cv2.getTrackbarPos('vlpink', self.name)
        vhpink = cv2.getTrackbarPos('vhpink', self.name)

        hwhite = cv2.getTrackbarPos('hwhite', self.name)
        slwhite = cv2.getTrackbarPos('slwhite', self.name)
        shwhite = cv2.getTrackbarPos('shwhite', self.name)
        vlwhite = cv2.getTrackbarPos('vlwhite', self.name)
        vhwhite = cv2.getTrackbarPos('vhwhite', self.name)

        lowp = np.array([(180 + (hpink - prange)) % 180, slpink, vlpink])
        highp = np.array([(hpink + prange) % 180, shpink, vhpink])
        loww = np.array([(180 + (hwhite - wrange)) % 180, slwhite, vlwhite])
        highw = np.array([(hwhite + wrange) % 180, shwhite, vhwhite])

        arr = [lowp, highp, loww, highw]
        return arr

    def get_threshold(self):
        th = cv2.getTrackbarPos('threshold', self.name)
        return th

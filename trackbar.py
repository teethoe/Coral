import cv2
import numpy as np


def nothing(x):
    pass


class TrackbarWindow:
    def __init__(self, name, img, **kwargs):
        self.namep = name + ' Pink'
        self.namew = name + ' White'
        self.img = img

        th = kwargs.get('th', None)

        cv2.namedWindow(self.namep)
        #cv2.imshow(name, self.img)

        cv2.createTrackbar('hpink', self.namep, 0, 180, nothing)
        cv2.createTrackbar('slpink', self.namep, 0, 255, nothing)
        cv2.createTrackbar('shpink', self.namep, 0, 255, nothing)
        cv2.createTrackbar('vlpink', self.namep, 0, 255, nothing)
        cv2.createTrackbar('vhpink', self.namep, 0, 255, nothing)

        cv2.namedWindow(self.namew)

        cv2.createTrackbar('hwhite', self.namew, 0, 180, nothing)
        cv2.createTrackbar('slwhite', self.namew, 0, 255, nothing)
        cv2.createTrackbar('shwhite', self.namew, 0, 255, nothing)
        cv2.createTrackbar('vlwhite', self.namew, 0, 255, nothing)
        cv2.createTrackbar('vhwhite', self.namew, 0, 255, nothing)

        if th:
            cv2.createTrackbar('threshold', self.namew, 0, 255, nothing)

    def get_values(self):
        hpink = cv2.getTrackbarPos('hpink', self.namep)
        slpink = cv2.getTrackbarPos('slpink', self.namep)
        shpink = cv2.getTrackbarPos('shpink', self.namep)
        vlpink = cv2.getTrackbarPos('vlpink', self.namep)
        vhpink = cv2.getTrackbarPos('vhpink', self.namep)

        hwhite = cv2.getTrackbarPos('hwhite', self.namew)
        slwhite = cv2.getTrackbarPos('slwhite', self.namew)
        shwhite = cv2.getTrackbarPos('shwhite', self.namew)
        vlwhite = cv2.getTrackbarPos('vlwhite', self.namew)
        vhwhite = cv2.getTrackbarPos('vhwhite', self.namew)

        arr = [hpink, slpink, shpink, vlpink, vhpink, hwhite, slwhite, shwhite, vlwhite, vhwhite]
        return arr

    def get_range(self, prange, wrange):
        hpink = cv2.getTrackbarPos('hpink', self.namep)
        slpink = cv2.getTrackbarPos('slpink', self.namep)
        shpink = cv2.getTrackbarPos('shpink', self.namep)
        vlpink = cv2.getTrackbarPos('vlpink', self.namep)
        vhpink = cv2.getTrackbarPos('vhpink', self.namep)

        hwhite = cv2.getTrackbarPos('hwhite', self.namew)
        slwhite = cv2.getTrackbarPos('slwhite', self.namew)
        shwhite = cv2.getTrackbarPos('shwhite', self.namew)
        vlwhite = cv2.getTrackbarPos('vlwhite', self.namew)
        vhwhite = cv2.getTrackbarPos('vhwhite', self.namew)

        lowp = np.array([(180 + (hpink - prange)) % 180, slpink, vlpink])
        highp = np.array([(hpink + prange) % 180, shpink, vhpink])
        loww = np.array([(180 + (hwhite - wrange)) % 180, slwhite, vlwhite])
        highw = np.array([(hwhite + wrange) % 180, shwhite, vhwhite])

        arr = [lowp, highp, loww, highw]
        return arr

    def get_threshold(self):
        th = cv2.getTrackbarPos('threshold', self.namew)
        return th

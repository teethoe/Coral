import cv2
import numpy as np


class Process2:
    def __init__(self, img):
        self.img = img
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.hpink = 0
        self.hwhite = 0
        self.lowp = 0
        self.highp = 0
        self.loww = 0
        self.highw = 0
        self.flipw = False

    def mask(self, kernel):
        if self.lowp[0] < self.highp[0]:
            maskp = cv2.inRange(self.hsv, self.lowp, self.highp)
        else:
            temp = self.lowp[0]
            self.lowp[0] = self.highp[0]
            self.highp[0] = temp
            maskp = cv2.bitwise_not(cv2.inRange(self.hsv, self.lowp, self.highp))
        if self.loww[0] < self.highp[0]:
            self.flipw = False
            maskw = cv2.inRange(self.hsv, self.loww, self.highw)
        else:
            self.flipw = True
            temp = self.loww[0]
            self.loww[0] = self.highw[0]
            self.highw[0] = temp
            maskw = cv2.bitwise_not(cv2.inRange(self.hsv, self.loww, self.highw))
        temp = maskp + maskw
        closing = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(opening, 1, 2)
        l = [0 for i in range(4)]
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= l[2] and h >= l[3]:
                l = [x, y, w, h]
        return l, opening

    def maskth(self, value, kernel):
        grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(grey, value, 255, cv2.THRESH_BINARY_INV)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(closing, 1, 2)
        l = [0 for i in range(4)]
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= l[2] and h >= l[3]:
                l = [x, y, w, h]
        return l, closing

    def res(self, img, mask):
        res = cv2.bitwise_and(img, img, mask=mask)
        #res = cv2.blur(res,(5,5))
        return res

    def bleached(self, res, kernel):
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        if not self.flipw:
            white = cv2.inRange(hsv, self.loww, self.highw)
        else:
            ret, mask = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
            npink = cv2.bitwise_not(cv2.inRange(hsv, self.loww, self.highw))
            white = cv2.bitwise_and(npink, npink, mask=mask)
        closing = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        return closing


def fix(maskb, maska):
    ha, wa = maska.shape[:2]
    # detect ORB features and compute descriptors
    orb_detector = cv2.ORB_create(5000)
    kpb, db = orb_detector.detectAndCompute(maskb, None)
    kpa, da = orb_detector.detectAndCompute(maska, None)
    # match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(db, da)
    # sort matches by score and take top matches
    matches.sort(key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)
    # extract location of good matches
    ptsb = np.zeros((no_of_matches, 2))
    ptsa = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        ptsb[i, :] = kpb[matches[i].queryIdx].pt
        ptsa[i, :] = kpa[matches[i].trainIdx].pt
    # find homography + perspective warping transformation
    homography, mask = cv2.findHomography(ptsb, ptsa, cv2.RANSAC)
    trans = cv2.warpPerspective(maskb, homography, (wa, ha))
    return trans

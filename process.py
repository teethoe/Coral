import cv2
import numpy as np


class Process:
    def __init__(self, img):
        self.img = img
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.hpink = 0
        self.hwhite = 0
        self.lowp = 0
        self.highp = 0
        self.loww = 0
        self.highw = 0

    def mask(self, kernel):
        if self.lowp[0] < self.highp[0]:
            maskp = cv2.inRange(self.hsv, self.lowp, self.highp)
        else:
<<<<<<< HEAD
            temp = self.lowp[0]
            self.lowp[0] = self.highp[0]
            self.highp[0] = temp
            maskp = cv2.bitwise_not(cv2.inRange(self.hsv, self.lowp, self.highp))
=======
            maskp = cv2.bitwise_not(cv2.inRange(self.hsv, self.highp, self.lowp))
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
            # oneArr = np.ones(self.hsv.shape[:2])
            # maskp = oneArr - cv2.inRange(self.hsv, self.lowp, self.highp)
        if self.loww[0] < self.highp[0]:
            maskw = cv2.inRange(self.hsv, self.loww, self.highw)
        else:
<<<<<<< HEAD
            temp = self.loww[0]
            self.loww[0] = self.highw[0]
            self.highw[0] = temp
            maskw = cv2.bitwise_not(cv2.inRange(self.hsv, self.loww, self.highw))
=======
            maskw = cv2.bitwise_not(cv2.inRange(self.hsv, self.highw, self.loww))
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
        temp = maskp + maskw
        closing = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        #opening = cv2.normalize(src=opening, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        contours, hierarchy = cv2.findContours(opening, 1, 2)
        l = [0 for i in range(4)]
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= l[2] and h >= l[3]:
                l = [x, y, w, h]
        x, y, w, h = l
        self.img = self.img[y:y + h, x:x + w]
<<<<<<< HEAD
        #mask = cv2.blur(opening[y:y + h, x:x + w],(5,5))
        mask = opening[y:y + h, x:x + w]
=======
        mask = cv2.blur(opening[y:y + h, x:x + w],(5,5))
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
        #mask[mask < 0] = 0
        return mask, self.img

    def res(self, mask):
        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        #res = cv2.blur(res,(5,5))
        return res

    def bleached(self, res, kernel):
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, self.loww, self.highw)
        closing = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        return closing


def fix(maskb, maska):
    ha, wa = maska.shape[:2]
    orb_detector = cv2.ORB_create(5000)
    kpb, db = orb_detector.detectAndCompute(maskb, None)
    kpa, da = orb_detector.detectAndCompute(maska, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(db, da)
    matches.sort(key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)
    pb = np.zeros((no_of_matches, 2))
    pa = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        pb[i, :] = kpb[matches[i].queryIdx].pt
        pa[i, :] = kpa[matches[i].trainIdx].pt
    homography, mask = cv2.findHomography(pb, pa, cv2.RANSAC)
    trans = cv2.warpPerspective(maskb, homography, (wa, ha))
    return trans

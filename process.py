import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from colourFunc import *
from sklearn.cluster import KMeans
from collections import Counter


class Process:
    def __init__(self, img):
        self.img = img
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.lowp = np.array([0 for i in range(3)])
        self.highp = np.array([0 for i in range(3)])
        self.loww = np.array([0 for i in range(3)])
        self.highw = np.array([0 for i in range(3)])
        self.hsv_colours = [[0 for i in range(3)] for j in range(3)]
        self.rgb_colours = [[0 for i in range(3)] for j in range(3)]
        self.counts = Counter()

    def cluster(self, r):
        mod = self.hsv.reshape(self.hsv.shape[0] * self.hsv.shape[1], 3)
        clf = KMeans(n_clusters=3)
        labels = clf.fit_predict(mod)
        self.counts = Counter(labels)
        self.hsv_colours = clf.cluster_centers_
        pink = [c for c in self.hsv_colours if c[0] > 150 and c[1] > 100][0]
        self.lowp = np.array([(pink[0]-r+180) % 180, 50, 0])
        self.highp = np.array([(pink[0]+r) % 180, 255, 255])

    def pie(self, r):
        self.cluster(r)
        norm_hsv = normalise_hsv(self.hsv_colours)
        self.rgb_colours = denormalise_rgb([hsv_to_rgb(norm_hsv[i]) for i in range(len(norm_hsv))])
        hex_colours = [rgb2hex(self.rgb_colours[i]) for i in range(len(self.rgb_colours))]
        plt.figure(figsize=(8, 6))
        plt.pie(self.counts.values(), labels=hex_colours, colors=hex_colours)
        plt.show()

    def mask(self, kernel):
        if self.lowp[0] < self.highp[0]:
            maskp = cv2.inRange(self.hsv, self.lowp, self.highp)
        else:
            temp = self.lowp[0]
            self.lowp[0] = self.highp[0]
            self.highp[0] = temp
            maskp = cv2.bitwise_not(cv2.inRange(self.hsv, self.lowp, self.highp))
            # oneArr = np.ones(self.hsv.shape[:2])
            # maskp = oneArr - cv2.inRange(self.hsv, self.lowp, self.highp)
        if self.loww[0] < self.highp[0]:
            maskw = cv2.inRange(self.hsv, self.loww, self.highw)
        else:
            temp = self.loww[0]
            self.loww[0] = self.highw[0]
            self.highw[0] = temp
            maskw = cv2.bitwise_not(cv2.inRange(self.hsv, self.loww, self.highw))
        temp = maskp + maskw
        closing = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
        kernel2 = np.ones((8, 8), np.uint8)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
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
        mask = opening[y:y + h, x:x + w]
        mask = cv2.blur(mask, (5, 5))
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

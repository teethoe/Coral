import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.cluster import KMeans
from collections import Counter
from process import Process


imgb = cv2.imread('./img/ref/new/1.jpg')
imgb = cv2.resize(imgb, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
hsv = cv2.cvtColor(imgb, cv2.COLOR_BGR2HSV)

kernel = np.ones((5, 5), np.uint8)


def bgr2hex(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[2]), int(colour[1]), int(colour[0]))


def rgb2hex(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[0]), int(colour[1]), int(colour[2]))


def bgr2rgb(colour):
    return [colour[2], colour[1], colour[0]]


def normalise_hsv(arr):
    norm = arr.copy()
    for i in range(len(norm)):
        norm[i][0] *= (1/180)
        for j in range(2):
            norm[i][j+1] *= (1/255)
    return norm


def normalise_rgb(arr):
    norm = arr.copy()
    for i in range(len(norm)):
        for j in range(3):
            norm[i][j] *= (1/255)
    return norm


def denormalise_hsv(norm):
    arr = norm.copy()
    for i in range(len(arr)):
        arr[i][0] *= 180
        for j in range(2):
            arr[i][j+1] *= 255
    return arr


def denormalise_rgb(norm):
    arr = norm.copy()
    for i in range(len(arr)):
        for j in range(3):
            arr[i][j] *= 255
    return arr


mod = hsv.reshape(hsv.shape[0] * hsv.shape[1], 3)
clf = KMeans(n_clusters=3)
labels = clf.fit_predict(mod)
counts = Counter(labels)
center_colours = clf.cluster_centers_
hsv_colours = center_colours
norm_hsv = normalise_hsv(hsv_colours)
rgb_colours = denormalise_rgb([hsv_to_rgb(norm_hsv[i]) for i in range(len(hsv_colours))])
#rgb_colours = [bgr2rgb(center_colours[i]) for i in range(len(center_colours))]
hex_colours = [rgb2hex(rgb_colours[i]) for i in range(len(rgb_colours))]
plt.figure(figsize=(8, 6))
plt.pie(counts.values(), labels=hex_colours, colors=hex_colours)
print(hsv_colours)
pink = [c for c in hsv_colours if c[0] > 150 and c[1] > 100][0]
print(pink)

pbef = Process(imgb)
pbef.lowp = np.array([(pink[0]-10+180)%180, 50, 100])
pbef.highp = np.array([(pink[0]+10)%180, 255, 255])
pbef.loww = np.array([90, 0, 150])
pbef.highw = np.array([120, 70, 255])
maskb, imgb = pbef.mask(kernel)
resb = pbef.res(maskb)

cv2.imshow('imgb', imgb)
cv2.imshow('maskb', maskb)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

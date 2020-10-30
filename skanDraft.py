import cv2
import numpy as np
from process import Process
import matplotlib.pyplot as plt
from skimage import morphology, filters
from skan import draw, skeleton_to_csgraph, Skeleton, summarize

kernel = np.ones((5,5),np.uint8)

imgb = cv2.imread('./img/Coral before.png', cv2.IMREAD_UNCHANGED)
#h, w = imgb.shape[:2]
#imgb = cv2.resize(imgb,(h,h), interpolation=cv2.INTER_CUBIC)
pbef = Process(imgb)
pbef.lowp = np.array([150,60,100])
pbef.highp = np.array([170,255,255])
pbef.loww = np.array([90,15,150])
pbef.highw = np.array([115,255,255])
maskb = pbef.mask(kernel)
resb = pbef.res(maskb)
maskb = cv2.blur(maskb, (5,5))


binaryb = maskb > filters.threshold_otsu(maskb)
skeletonb = morphology.skeletonize(binaryb)
fig, ax = plt.subplots()
draw.overlay_skeleton_2d(maskb, skeletonb, dilate=1, axes=ax)

#graphb = csgraph_from_masked(binaryb)
#plt.imshow(graphb)
gb, cb, db = skeleton_to_csgraph(skeletonb)
draw.overlay_skeleton_networkx(gb, cb, image=maskb)
branch_datab = summarize(Skeleton(skeletonb))
dfb = branch_datab.loc[branch_datab['branch-type']==1]

#dfb.to_csv(r'./before.csv')
draw.overlay_euclidean_skeleton_2d(maskb, branch_datab, skeleton_color_source='branch-type');


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
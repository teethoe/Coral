import cv2
import matplotlib.pyplot as plt
from skimage import morphology, filters
from skan import draw, skeleton_to_csgraph, Skeleton, summarize


class skn:
    def __init__(self, mask):
        self.mask = mask

    def skan(self, n):
        blur = cv2.blur(self.mask, (5, 5))
        binary = blur > filters.threshold_otsu(blur)
        skeleton = morphology.skeletonize(binary)
        ax = plt.subplot(2, 2, 2*n-1)
        draw.overlay_skeleton_2d(blur, skeleton, dilate=1, axes=ax)
        branch_data = summarize(Skeleton(skeleton))
        ax = plt.subplot(2, 2, 2*n)
        draw.overlay_euclidean_skeleton_2d(blur, branch_data, skeleton_color_source='branch-type', axes=ax)
        df1 = branch_data.loc[branch_data['branch-type'] == 1]
        return df1

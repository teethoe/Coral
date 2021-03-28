import cv2
<<<<<<< HEAD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
=======
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e


class Change:
    def __init__(self, img, maskb, maska):
        self.img = img
        ret, self.maskb = cv2.threshold(maskb, 1, 255, cv2.THRESH_BINARY)
        ret, self.maska = cv2.threshold(maska, 1, 255, cv2.THRESH_BINARY)
        # BGR
        self.green = (0, 255, 0)
        self.yellow = (0, 255, 255)
        self.red = (0, 0, 255)
        self.blue = (255, 0, 0)
        self.change = img.copy()

    def growth_death(self, kernel):
        growth = self.maska - self.maskb
        death = self.maskb - self.maska
<<<<<<< HEAD
        cv2.imshow('growth', growth)
        cv2.imshow('death', death)
        cv2.waitKey(0)
=======
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
        arr = [growth, death]
        colours = [self.green, self.yellow]
        for j in range(len(arr)):
            ret, arr[j] = cv2.threshold(arr[j], 1, 255, cv2.THRESH_BINARY)
            arr[j] = cv2.morphologyEx(arr[j], cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(arr[j], 1, 2)
            for i in range(len(contours)):
                cnt = contours[i]
                x, y, w, h = cv2.boundingRect(cnt)
<<<<<<< HEAD
                if w > 15 and h > 15:
=======
                if w>15 and h>15:
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
                    self.change = cv2.rectangle(self.change, (x,y), (x+w,y+h), colours[j], 2)
        return arr

    def bleach_recover(self, whiteb, whitea, growth, death, kernel):
        bleach = (whitea - whiteb) - growth
        recover = (whiteb - whitea) - death
        arr = [bleach, recover]
        colours = [self.red, self.blue]
        for j in range(len(arr)):
            ret, arr[j] = cv2.threshold(arr[j], 1, 255, cv2.THRESH_BINARY)
            arr[j] = cv2.morphologyEx(arr[j], cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(arr[j], 1, 2)
            for i in range(len(contours)):
                cnt = contours[i]
                x, y, w, h = cv2.boundingRect(cnt)
<<<<<<< HEAD
                if w > 15 and h > 15:
=======
                if w>15 and h>15:
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
                    self.change = cv2.rectangle(self.change, (x,y), (x+w,y+h), colours[j], 2)
        return arr

    def final(self):
        return self.change
<<<<<<< HEAD


def plot(imgb, imga, final):
    fig, axs = plt.subplots(1, 3, dpi=400)
    rgb_imgb = imgb[..., ::-1]
    axs[0].imshow(rgb_imgb)
    axs[0].set_title('Before')
    axs[0].axis('off')
    rgb_imga = imga[..., ::-1]
    axs[1].imshow(rgb_imga)
    axs[1].set_title('After')
    axs[1].axis('off')
    rgb_final = final[..., ::-1]
    axs[2].imshow(rgb_final)
    axs[2].set_title('Final')
    axs[2].axis('off')
    plt.show()
=======
>>>>>>> 54fc90ae7764bd086d4603aaa049854a1462df7e
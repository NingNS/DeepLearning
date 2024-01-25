import cv2

dataroot = 'test'
saveroot = 'enhance_images'
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

bgr = None


def get_rgb_(i):
    global bgr
    img = cv2.imread(dataroot + "/%d.jpg" % i, cv2.IMREAD_COLOR)
    bgr = img
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('HSV', HSV)
    H, S, V = cv2.split(HSV)
    # cv2.imshow('S', V)
    # cv2.waitKey(0)
    return H, S, V


def HE(i):
    img = cv2.imread(dataroot + "/%d.jpg" % i, 0)
    equ = cv2.equalizeHist(img)  # 输入为灰度图
    res = np.hstack((img, equ))  # stacking images side-by-side
    cv2.imshow(dataroot + "/%d.jpg" % i, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def CLAHE(img, i):
    # #返回彩色图像
    # image = cv2.imread('meiss.png', cv2.IMREAD_COLOR)
    # #将彩色图像分割为b,g,r3个通道
    # b, g, r = cv2.split(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # 剪辑限制(对比度阈值)为0.01，图块数量为8*8,有点问题
    clahe = cv2.createCLAHE(clipLimit=1.81, tileGridSize=(8, 8))
    l, a, b = cv2.split(gray_image)
    L = clahe.apply(l)
    image = cv2.merge([L, a, b])
    new2_rbg_color = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    cv2.imwrite(saveroot + "/%d.jpg" % i, new2_rbg_color, [cv2.IMWRITE_JPEG_QUALITY, 50])


def gama_(img):
    # 默认gamma值为1.0，默认不变化
    def adjust_gamma(image, gamma=2.2):
        invgamma = 1 / gamma
        brighter_image = np.array(np.power((image / 255), invgamma) * 255, dtype=np.uint8)
        return brighter_image

    # gamma大于1，变亮
    img_gamma1 = adjust_gamma(img, gamma=2.2)
    # gamma小于1，变暗
    img_gamma2 = adjust_gamma(img, gamma=0.5)
    return img_gamma1


if __name__ == '__main__':
    for i in range(1, 16):
        H, S, V = get_rgb_(i)
        new_V = gama_(V)
        new_img_hsv = cv2.merge((H, S, new_V))
        new_bgr_color = cv2.cvtColor(new_img_hsv, cv2.COLOR_HSV2BGR)
        b_x, g_x, r_x = cv2.split(bgr)
        b1_x, g1_x, r1_x = cv2.split(new_bgr_color)
        # print(b1_x / b_x)
        # Gxy = list(np.concatenate(np.nan_to_num((b1_x / b_x)).reshape((-1, 1), order="F")))
        # print(sum(map(sum, np.nan_to_num(b1_x / b_x))) / len(Gxy))
        # print(len(Gxy))
        # w,h,_ = new_bgr_color.shape
        # lab = np.zeros((w,h,3))
        # for i in range(w):
        #     for j in range(h):
        #         Lab = RGB2LAB(new_bgr_color[i,j])
        #         lab[i,j] = (Lab[0],Lab[1],Lab[2])

        # l,a,b = cv2.split(lab)

        CLAHE(new_bgr_color, i)

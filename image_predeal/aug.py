import cv2
import numpy as np
import os
import random
'''
图像增强，img_augmentation_test测试效果
img_augmentation为扩中数据集
'''
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# dimming
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


def rotate(image, angle=3, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image

def img_augmentation_test(path):
    img = cv2.imread(path)
    img_flip = cv2.flip(img, 1)  # flip
    img_rotation = rotate(img)  # rotation

    img_noise1 = SaltAndPepper(img, 0.3)
    img_noise2 = addGaussianNoise(img, 0.3)

    img_brighter = brighter(img)
    img_darker = darker(img)

    cv2.imshow("flip", img_flip)
    cv2.imshow("rotation", img_rotation)
    cv2.imshow("noise1", img_noise1)
    cv2.imshow("noise2", img_noise2)
    cv2.imshow("brighter", img_brighter)
    cv2.imshow("darker", img_darker)
    cv2.waitKey(0)
def img_augmentation(filepath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.isfile(child):
            img = cv2.imread(child)
            for i in range(10):
                img_flip = cv2.flip(img, 1)  # flip
                cv2.imwrite(child+(str)(i)+"_"+"img_flip"+".png",img_flip)
                img_rotation = rotate(img)  # rotation
                cv2.imwrite(child + (str)(i) + "_" + "img_rotation" + ".png", img_rotation)
                img_noise1 = SaltAndPepper(img, random.random(0,10)*0.1)
                cv2.imwrite(child + (str)(i) + "_" + "img_noise1" + ".png", img_noise1)
                img_noise2 = addGaussianNoise(img, random.random(0,10)*0.1)
                cv2.imwrite(child + (str)(i) + "_" + "img_noise2" + ".png", img_noise2)
                img_brighter = brighter(img)
                cv2.imwrite(child + (str)(i) + "_" + "img_brighter" + ".png", img_brighter)
                img_darker = darker(img)
                cv2.imwrite(child + (str)(i) + "_" + "img_darker" + ".png", img_darker)

if __name__ == "__main__":
    img_augmentation_test("../dataset2/number/block2.png")#测试结果
    img_augmentation("../dataset/image_0")#数据集扩充
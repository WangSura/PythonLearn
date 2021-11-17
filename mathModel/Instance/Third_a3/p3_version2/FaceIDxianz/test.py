#！/usr/bin/env python
# -*- coding：utf-8 -*-
# author: haotian time:2019/8/20

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys



def CatchUsbVideo(window_name):
    grey = cv2.imread("./Trained_Face/pic3.jpg")
    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)


    # grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)

    # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=1, minSize=(32, 32))
    if len(faceRects) > 0:  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            cv2.rectangle(grey, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            # cv2.rectangle(grey, (x + 10, y + 10), (x - 10 + w, y - 10 + h), color, 2)

    # 显示图像
    # cv2.imshow(window_name, grey)
    # cv2.waitKey()
    cv2.imwrite('1.png',grey)
    # grey = Image.open('1.png')
    # grey = grey.resize((334, 501))
    # plt.ion()
    # plt.imshow(grey)
    # # plt.pause(0.1)
    # plt.close()

    # cv2.destroyAllWindows()

CatchUsbVideo('window_name')
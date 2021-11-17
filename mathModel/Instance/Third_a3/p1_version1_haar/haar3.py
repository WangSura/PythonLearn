import cv2
import numpy as np
import sys
import os
import glob
import numpy
from skimage import io


# 指定图片的人脸识别然后存储
img = cv2.imread(
    "D:/program/pythonPractice/mathModel/Instance/Third_a3/Origin/p3.jpg")
color = (0, 255, 0)
cv2.imshow("Origin Faces!", img)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

classfier = cv2.CascadeClassifier(
    "D:\\program\\python391\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml")


faceRects = classfier.detectMultiScale(
    grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects) > 0:  # 大于0则检测到人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        x, y, w, h = faceRect
        cv2.rectangle(img, (x - 10, y - 10), (x + w + 10,
                                              y + h + 10), color, 3)  # 5控制绿色框的粗细


# 写入图像
cv2.imwrite(
    'D:/program/pythonPractice/mathModel/Instance/Third_a3/Origin/output3.jpg', img)
cv2.imshow("Find Faces!", img)
cv2.waitKey(0)

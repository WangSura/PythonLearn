import cv2
import numpy as np
import sys
import os
import glob
import numpy
from skimage import io


# 指定图片的人脸识别然后存储
img = cv2.imread(
    "D:/program/pythonPractice/mathModel/Instance/Third_a3/Origin/p1.jpg")
color = (0, 255, 0)
# cv2.imshow("Origin Faces!", img)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(
    'D:/program/pythonPractice/mathModel/Instance/Third_a3/Origin2/p1.jpg', grey)


classfier = cv2.CascadeClassifier(
    "D:\\program\\python391\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml")

num = 0
path = "D:/program/pythonPractice/mathModel/Instance/Third_a3/Origin2"
faceRects = classfier.detectMultiScale(
    grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects) > 0:  # 大于0则检测到人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        x, y, w, h = faceRect

        # 将当前帧保存为图片
        img_name = '%s/%d.jpg' % (path, num)

        cv2.rectangle(img, (x - 10, y - 10), (x + w + 10,
                                              y + h + 10), color, 3)  # 5控制绿色框的粗细
        image_alone = img[y - 10:y + h + 10, x - 10:x + w + 10]
        cv2.imwrite(img_name, image_alone)
        num += 1

# 写入图像
cv2.imwrite(
    'D:/program/pythonPractice/mathModel/Instance/Third_a3/Origin2/output1.jpg', img)

cv2.waitKey(0)

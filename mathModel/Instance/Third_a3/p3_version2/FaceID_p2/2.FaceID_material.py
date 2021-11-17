#！/usr/bin/env python
# -*- coding：utf-8 -*-
# author: haotian time:2019/8/19
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
# !/practice/Study_Test python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/23 21:19
# @Author  : yb.w
# @File    : imageface.py
# @define  : 检测图片中的人脸,用矩形框标出
# OpenCv   :
# 使用OpenCV自带库参数数据


# 1.导入库
import cv2

# 2.加载图片，加载模型

# 待检测的图片路径
imagepath = r'./Trained_Face/pic3.jpg'
# 获取训练好的人脸的参数数据，这里直接使用默认值
pathf = r'./haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(pathf)
# 读取图片
image = cv2.imread(imagepath)

# 3.对图片灰度化处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4.检测人脸，探测图片中的人脸
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.15,
    minNeighbors=5,
    minSize=(5, 5),
    # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)
print("发现{0}个人脸!".format(len(faces)))

# 5.标记人脸
for (x, y, w, h) in faces:
    # 1.原始图片 2.人脸坐标原点 3.标记的高度 4，线的颜色 5，线宽
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

# 6.显示图片
cv2.imshow("Find Faces!", image)
cv2.imwrite('1.jpg',image)

# 7.暂停窗口
cv2.waitKey()

# 8.销毁窗口
cv2.destroyAllWindows()


'''
# import cv2
# import numpy as np
#
# img = cv2.imread("./Trained_Face/附件_图2.jpg")
# emptyImage = np.zeros(img.shape, np.uint8)
#
# emptyImage2 = img.copy()
#
# emptyImage3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("EmptyImage3", emptyImage3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''





# def CatchUsbVideo(window_name):
#     grey = cv2.imread("./Trained_Face/pic1.jpg")
#     # 告诉OpenCV使用人脸识别分类器
#     classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
#
#     # 识别出人脸后要画的边框的颜色，RGB格式
#     color = (0, 255, 0)
#
#
#     # grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
#
#     # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
#     faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=1, minSize=(32, 32))
#     if len(faceRects) > 0:  # 大于0则检测到人脸
#         for faceRect in faceRects:  # 单独框出每一张人脸
#             x, y, w, h = faceRect
#             cv2.rectangle(grey, (x - 10, y - 10), (x + 10, y + 10), color, 2)
#             cv2.rectangle(grey, (x + 10, y - 10), (x - 10, y + 10), color, 2)
#
#
#     # 显示图像
#     # cv2.imshow(window_name, grey)
#     # cv2.waitKey()
#     cv2.imwrite('1.png',grey)
#     grey = Image.open('1.png')
#     grey = grey.resize((334, 501))
#     plt.ion()
#     plt.imshow(grey)
#     # plt.pause(0.1)
#     plt.close()
#
#     # c = cv2.waitKey(10)
#     # if c & 0xFF == ord('q'):
#     #     break
#     # plt.imshow(grey)
#         # 释放摄像头并销毁所有窗口
#     # cap.release()
#     # cv2.destroyAllWindows()
#
# CatchUsbVideo('window_name')
# # if __name__ == '__main__':
# #     if len(sys.argv) != 1:
# #         print("Usage:%s camera_id\r\n" % (sys.argv[0]))
# #     else:
#         # CatchUsbVideo("识别人脸区域", 0)

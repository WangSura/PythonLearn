# 1.导入库
import cv2
# 获取训练好的人脸的参数数据
face_cascade_path = ".\\haarcascades\\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade_path = ".\\haarcascades\\haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
nose_cascade_path = ".\\haarcascades\\haarcascade_mcs_nose.xml"
nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
mouth_cascade_path = ".\\haarcascades\\haarcascade_mcs_mouth.xml"
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

# 2.引入图片
input_image1_path = './Trained_Face/pic1.jpg'
input_image2_path = './Trained_Face/pic2.jpg'
input_image3_path = './Trained_Face/pic3.jpg'

# 3.读取图片
image = cv2.imread(input_image2_path)
# 4.灰度化

# 5.标记人脸
# xxx_cascade.detectMultiScale()
    # 1.要检测的输入图像
    # 2.每次图像尺寸减小的比例
    # 3.每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸
    # 4.要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
    # CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，因此这些区域通常不会是人脸所在区域；
    # 5.目标的最小尺寸
    # 6.目标的最大尺寸

faces = face_cascade.detectMultiScale(image,1.1,15,cv2.CASCADE_SCALE_IMAGE,(50,50))
print("发现{0}个人脸!".format(len(faces)))
count = 0
for (x, y, w, h) in faces:
    print('第'+str(count+1)+'个人脸中心坐标' + ' x: '+str(x)+' y: '+str(y))
    print('宽度: ' + str(w))
#     # 1.原始图片 2.人脸坐标原点 3.标记的高度 4，线的颜色 5，线宽
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5,8,0)

    # 人脸切割
    roi_gray = image[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100))
    num = 1
    for (ex, ey, ew, eh) in eyes:
        print('第' + str(num) + '个眼睛中心坐标'+' x: '+str(ex)+' y: '+str(ey))
        # 1.原始图片 2.人脸坐标原点 3.标记的高度 4，线的颜色 5，线宽
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
        print('宽度: ' + str(ew) + ' 长度: ' + str(eh))
        num = num+1


    num=1
    noses = nose_cascade.detectMultiScale(roi_gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (125, 125))
    for (nx, ny, nw, nh) in noses:
        print('第' + str(num) + '个鼻子中心坐标'+' x: '+str(nx)+' y: '+str(ny))
        print('宽度: ' + str(nw) + ' 长度: ' + str(nh))
        # 1.原始图片 2.人脸坐标原点 3.标记的高度 4，线的颜色 5，线宽
        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 5)
        num = num + 1


    num = 1
    mouths = mouth_cascade.detectMultiScale(roi_gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (150, 150))
    for (mx, my, mw, mh) in mouths:
        print('第' + str(num) + '个嘴巴中心坐标'+' x: '+str(mx)+' y: '+str(my))
        print('宽度: ' + str(mw) + ' 长度: ' + str(mh))
        # 1.原始图片 2.人脸坐标原点 3.标记的高度 4，线的颜色 5，线宽
        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 0), 5)
        num = num + 1

    print('\n')

    count = count+1

# 6.显示图片
cv2.imshow("Find Faces!", image)
cv2.imwrite('1.jpg',image)

# 7.暂停窗口
cv2.waitKey()

# 8.销毁窗口
cv2.destroyAllWindows()

import cv2

face_cascade_path = "D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/haarcascades/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade_path = "D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/haarcascades/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
nose_cascade_path = "D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/haarcascades/haarcascade_mcs_nose.xml"
nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
mouth_cascade_path = "D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/haarcascades/haarcascade_mcs_mouth.xml"
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)


input_image0_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/p0.jpg'
input_image1_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/pic1.jpg'
input_image2_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/pic2.jpg'
input_image3_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/pic3.jpg'
input_image4_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/4.jpg'
input_image5_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/5.jpg'
input_image6_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/6.jpg'
input_image7_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/7.jpg'
input_image8_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/8.jpg'
input_image9_path = 'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/9.jpg'


image = cv2.imread(input_image2_path)

color = (0, 255, 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    image, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
print("??????{0}?????????!".format(len(faces)))
count = 0
for (x, y, w, h) in faces:
    print('???'+str(count+1)+'?????????????????????' + ' x: '+str(x)+' y: '+str(y))
    print('??????: ' + str(w))

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5, 8, 0)

    # ????????????
    roi_gray = image[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(
        roi_gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100))
    num = 1
    for (ex, ey, ew, eh) in eyes:
        print('???' + str(num) + '?????????????????????'+' x: '+str(ex)+' y: '+str(ey))

        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
        print('??????: ' + str(ew) + ' ??????: ' + str(eh))
        num = num+1

    num = 1
    noses = nose_cascade.detectMultiScale(
        roi_gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (125, 125))
    for (nx, ny, nw, nh) in noses:
        print('???' + str(num) + '?????????????????????'+' x: '+str(nx)+' y: '+str(ny))
        print('??????: ' + str(nw) + ' ??????: ' + str(nh))

        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 5)
        num = num + 1

    num = 1
    mouths = mouth_cascade.detectMultiScale(
        roi_gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (150, 150))
    for (mx, my, mw, mh) in mouths:
        print('???' + str(num) + '?????????????????????'+' x: '+str(mx)+' y: '+str(my))
        print('??????: ' + str(mw) + ' ??????: ' + str(mh))

        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 0), 5)
        num = num + 1

    print('\n')

    count = count+1


cv2.imshow("Find Faces!", image)
cv2.imwrite(
    'D:\program\pythonPractice\mathModel\Instance\Third_a3\p3_version2\FaceID\Trained_Face\ha2beter.jpg', image)


cv2.waitKey()


cv2.destroyAllWindows()

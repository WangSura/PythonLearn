import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import cv2
from PIL import Image
from imblearn.over_sampling import SMOTE

IMAGE_SIZE = 224


def LoadData():
    data = []
    label = []
    num = 20
    path_cwd = "D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/train/"
    for j in range(1, 4):
        path = path_cwd + 's' + str(j)
        for number in range(num):
            path_full = path + '/' + str(number) + '.png'
            image = Image.open(path_full).convert('L')
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            img = np.reshape(image, (1, IMAGE_SIZE*IMAGE_SIZE))
            data.extend(img)
        label.extend(np.ones(num, dtype=np.int) * j)
    data = np.reshape(data, (num*j, IMAGE_SIZE*IMAGE_SIZE))
    return np.matrix(data), np.matrix(label).T


def svm(trainDataSimplified, trainLabel, testDataSimplified):
    clf3 = SVC(C=0.001, gamma=25.0)
    sm = SMOTE(random_state=42)
    trainDataSimplified, trainLabel = sm.fit_sample(
        trainDataSimplified, trainLabel.ravel())
    clf3.fit(trainDataSimplified, trainLabel)
    return clf3.predict(testDataSimplified)


def knn(neighbor, traindata, trainlabel, testdata):
    neigh = KNeighborsClassifier(n_neighbors=neighbor)
    sm = SMOTE(random_state=42)
    traindata, trainlabel = sm.fit_sample(traindata, trainlabel.ravel())
    neigh.fit(traindata, trainlabel)
    return neigh.predict(testdata)


if __name__ == '__main__':

    Data, Label = LoadData()
    pca = PCA(0.9, True, True)
    trainDataS = pca.fit_transform(Data)

    color = (0, 255, 0)

    cap = cv2.VideoCapture(0)

    cascade_path = "D:/program/python391/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
    i = 0

    while i == 0:

        frame = cv2.imread(
            'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/Trained_Face/pic3.jpg')
        frame_gray = frame

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(cascade_path)

        faceRects = cascade.detectMultiScale(
            frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                m = frame_gray[y - 10: y + h + 10, x - 10: x + w + 10]

                top, bottom, left, right = (0, 0, 0, 0)
                image = m

                h, w = image.shape

                longest_edge = max(h, w)

                if h < longest_edge:
                    dh = longest_edge - h
                    top = dh // 2
                    bottom = dh - top
                elif w < longest_edge:
                    dw = longest_edge - w
                    left = dw // 2
                    right = dw - left
                else:
                    pass

                BLACK = [0]

                constant = cv2.copyMakeBorder(
                    image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

                image = cv2.resize(constant, (IMAGE_SIZE, IMAGE_SIZE))
                img_test = np.reshape(image, (1, IMAGE_SIZE * IMAGE_SIZE))
                testDataS = pca.transform(img_test)
                result = svm(trainDataS, Label, testDataS)

                result = knn(31, trainDataS, Label, testDataS)
                faceID = result[0]

                if faceID == 1:
                    cv2.rectangle(frame, (x - 10, y - 10),
                                  (x + w + 10, y + h + 10), color, thickness=5)

                    cv2.putText(frame, 'YYQX',
                                (x + 30, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (255, 0, 255),
                                5)
                elif faceID == 2:
                    cv2.rectangle(frame, (x - 10, y - 10),
                                  (x + w + 10, y + h + 10), color, thickness=5)

                    cv2.putText(frame, 'WY',
                                (x + 30, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (255, 0, 255),
                                5)
                elif faceID == 3:
                    cv2.rectangle(frame, (x - 10, y - 10),
                                  (x + w + 10, y + h + 10), color, thickness=5)

                    cv2.putText(frame, 'WJK',
                                (x + 30, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (255, 0, 255),
                                5)
                else:
                    cv2.rectangle(frame, (x - 10, y - 10),
                                  (x + w + 10, y + h + 10), color, thickness=2)

                    cv2.putText(frame, 'NONE',
                                (x + 30, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                10,
                                (255, 0, 255),
                                5)

        i = 1

        cv2.imwrite(
            'D:/program/pythonPractice/mathModel/Instance/Third_a3/p3_version2/FaceID/haha.jpg', frame)
        cv2.imshow("find me", frame)
        cv2.waitKey()
    cv2.destroyAllWindows()

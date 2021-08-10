import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from sklearn.ensemble import IsolationForest
from lof_base import localoutlierfactor, plot_lof, lof


def lof(data, predict=None, k=5, method=1, plot=False):
    import pandas as pd

    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
    # 计算 LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    if plot == True:
        plot_lof(predict, method)
    # 根据阈值划分离群点与正常点
    outliers = predict[predict['local outlier factor']
                       > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <=
                      method].sort_values(by='local outlier factor')
    return outliers, inliers


posi = pd.read_excel(r'D:\program\pythonPractice\数模\数据清洗\离散值处理\已结束项目任务数据.xls')
lon = np.array(posi["任务gps经度"][:])  # 经度
lat = np.array(posi["任务gps 纬度"][:])  # 纬度
A = list(zip(lat, lon))  # 按照纬度-经度匹配

posi = pd.read_excel(r'D:\program\pythonPractice\数模\数据清洗\离散值处理\会员信息数据.xlsx')
lon = np.array(posi["会员位置(GPS)经度"][:])  # 经度
lat = np.array(posi["会员位置(GPS)纬度"][:])  # 纬度
B = list(zip(lat, lon))  # 按照纬度-经度匹配

# 获取会员对任务密度，取第5邻域，阈值为5（LOF大于5认为是离群值）
outliers3, inliers3 = lof(A, B, k=5, method=5)
plt.figure('k=5')
outliers3, inliers3 = lof(A, B, k=5, method=5)
plt.scatter(np.array(B)[:, 0], np.array(B)[:, 1], s=10, c='b', alpha=0.5)
plt.scatter(np.array(A)[:, 0], np.array(A)[:, 1], s=10, c='green', alpha=0.3)
plt.scatter(outliers3[0], outliers3[1], s=10 +
            outliers3['local outlier factor'] * 20, c='r', alpha=0.2)
plt.title('k = 5, method = 5')
plt.savefig('D:\program\pythonPractice\数模\数据清洗\离散值处理\lof_base2.png')  # 保存图片
plt.show()

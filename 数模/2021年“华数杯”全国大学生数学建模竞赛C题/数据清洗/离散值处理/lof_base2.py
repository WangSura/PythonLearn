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


posi = pd.read_excel(r'D:\program\pythonPractice\数模\2021年“华数杯”全国大学生数学建模竞赛C题\数据清洗\离散值处理\处理方便机器学习1.xlsx')
lat = np.array(posi["a1"][:])  # 纬度
A = list(zip(lat, np.zeros_like(lat)))  # 按照纬度-经度匹配

posi = pd.read_excel(r'D:\program\pythonPractice\数模\2021年“华数杯”全国大学生数学建模竞赛C题\数据清洗\离散值处理\处理方便机器学习1.xlsx')
lon = np.array(posi["a2"][:])  # 经度
B = list(zip(lon, np.zeros_like(lon)))  # 按照纬度-经度匹配

# 获取会员对任务密度，取第5邻域，阈值为5（LOF大于5认为是离群值）
outliers3, inliers3 = lof(A, B, k=3, method=5)
plt.figure('k=3')
outliers3, inliers3 = lof(A, B, k=3, method=5)
plt.scatter(np.array(B)[:, 0], np.array(B)[:, 1], s=10, c='b', alpha=0.5)
plt.scatter(np.array(A)[:, 0], np.array(A)[:, 1], s=10, c='green', alpha=0.3)
plt.scatter(outliers3[0], outliers3[1], s=10 +
            outliers3['local outlier factor'] * 20, c='r', alpha=0.2)
plt.title('k = 3, method = 5')
plt.savefig('D:\program\pythonPractice\数模\数据清洗\离散值处理\lof_base2.png')  # 保存图片
plt.show()

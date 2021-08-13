import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from sklearn.ensemble import IsolationForest


posi = pd.read_excel(r'D:\program\pythonPractice\数模\数据清洗\离散值处理\已结束项目任务数据.xls')
lon = np.array(posi["任务gps经度"][:])  # 经度
lat = np.array(posi["任务gps 纬度"][:])  # 纬度
A = list(zip(lat, lon))  # 按照纬度-经度匹配

posi = pd.read_excel(r'D:\program\pythonPractice\数模\数据清洗\离散值处理\会员信息数据.xlsx')
lon = np.array(posi["会员位置(GPS)经度"][:])  # 经度
lat = np.array(posi["会员位置(GPS)纬度"][:])  # 纬度
B = list(zip(lat, lon))  # 按照纬度-经度匹配


clf = LOF(n_neighbors=2)
res = clf.fit_predict(A)
print(res)
print(clf.negative_outlier_factor_)
xx, yy = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
'''
如果 X = [[-1.1], [0.2], [100.1], [0.3]]
[ 1  1 -1  1]
[ -0.98214286  -1.03703704 -72.64219576  -0.98214286]
 
如果 X = [[-1.1], [0.2], [0.1], [0.3]]
[-1  1  1  1]
[-7.29166666 -1.33333333 -0.875      -0.875     ]
 
如果 X = [[0.15], [0.2], [0.1], [0.3]]
[ 1  1  1 -1]
[-1.33333333 -0.875      -0.875      -1.45833333]
'''

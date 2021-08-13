# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as lof
# 构造训练样本
n_samples = 200  # 样本总数
outliers_fraction = 0.25  # 异常样本比例
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)


posi = pd.read_excel(r'D:\program\pythonPractice\数模\数据清洗\离散值处理\已结束项目任务数据.xls')
lon = np.array(posi["任务gps经度"][:])  # 经度
lat = np.array(posi["任务gps 纬度"][:])  # 纬度
A = list(zip(lat, lon))  # 按照纬度-经度匹配


# fit the model
clf = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
y_pred = clf.fit_predict(A)
scores_pred = clf.negative_outlier_factor_
threshold = stats.scoreatpercentile(
    scores_pred, 100 * outliers_fraction)  # 根据异常样本比例，得到阈值，用于绘图

# plot the level sets of the decision function
xx, yy = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))
# 类似scores_pred的值，值越小越有可能是异常点
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Local Outlier Factor (LOF)")
# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
             cmap=plt.cm.Blues_r)  # 绘制异常点区域，值从最小的到阈值的那部分
a = plt.contour(xx, yy, Z, levels=[threshold],
                linewidths=2, colors='red')  # 绘制异常点区域和正常点区域的边界
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],
             colors='palevioletred')  # 绘制正常点区域，值从阈值到最大的那部分

b = plt.scatter(A[:-n_outliers, 0], A[:-n_outliers, 1], c='white',
                s=20, edgecolor='k')
c = plt.scatter(A[-n_outliers:, 0], A[-n_outliers:, 1], c='black',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-7, 7))
plt.ylim((-7, 7))
plt.legend([a.collections[0], b, c],
           ['learned decision function', 'true inliers', 'true outliers'],
           loc="upper left")
plt.show()

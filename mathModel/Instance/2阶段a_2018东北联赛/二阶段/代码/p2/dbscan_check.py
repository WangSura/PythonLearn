from matplotlib.colors import Colormap
from sklearn import metrics
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import numpy as np
#from matplotlib import cm
import matplotlib.cm as cm
# 导入数据
beer = pd.read_excel(
    'D:/program/pythonPractice/mathModel/Instance/2阶段a_2018东北联赛/二阶段/A2题附件漏损速率.xls')
X = beer[["Speed"]]
# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=0.53, min_samples=68).fit(X)  # 36行

labels = db.labels_

beer['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
beer.sort_values('cluster_db')  # 排序数据集
beer.to_excel(
    'D:/program/pythonPractice/mathModel/Instance/2阶段a_2018东北联赛/二阶段/p2_dbscan4_3cu.xls')
# 注：cluster列是kmeans聚成3类的结果；cluster2列是kmeans聚类成2类的结果；scaled_cluster列是kmeans聚类成3类的结果（经过了数据标准化）
# 查看根据DBSCAN聚类后的分组统计结果（均值）
print(beer.groupby('cluster_db').mean())
# 画出在不同两个指标下样本的分布情况
# 删除[] 改为c=beer.cluster_db
# c=beer.cluster_db.tolist()
plt.figure(num=1)
print(pd.plotting.scatter_matrix(
    X, c=beer.cluster_db, figsize=(10, 10), s=100, alpha=.8, cmap=mglearn.cm3, marker='0', hist_kwds={'bins': 50}))
plt.show()

# plt.colorbar(pic)

# 我们可以从上面这个图里观察聚类效果的好坏，但是当数据量很大，或者指标很多的时候，观察起来就会非常麻烦。
# 就是下面这个函数可以计算轮廓系数（sklearn真是一个强大的包）
score = metrics.silhouette_score(X, beer.cluster_db)
print(score)

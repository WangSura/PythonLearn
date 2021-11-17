from warnings import filterwarnings
from sklearn.decomposition import PCAplt.style.use('fivethirtyeight')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.offline as py
from plotly.subplots import make_subplots
from plotly import tools
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
pyo.init_notebook_mode()
filterwarnings('ignore')
with open('HR_data.csv') as f:
df = pd.read_csv(f, usecols=['satisfaction_level', 'last_evaluation', 'number_project',
                             'average_montly_hours', 'time_spend_company', 'Work_accident',
                             'promotion_last_5years'])
f.close()
# 标准化
scaler = StandardScaler()
scaler.fit(df)
X_scale = scaler.transform(df)
df_scale = pd.DataFrame(X_scale, columns=df.columns)
df_scale.head()
# 特征降维
# 需要确定适当的主成分数量。3个主成分似乎占了大约75%的方差
pca = PCA(n_components=7)
pca.fit(df_scale)
variance = pca.explained_variance_ratio_
var = np.cumsum(np.round(variance, 3)*100)
plt.figure(figsize=(12, 6))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(0, 100.5)
plt.plot(var)
# 第一个主成分占到与原始数据集方差的26%。在本文的其余部分中，我们将使用“pca_df”数据框架
pca = PCA(n_components=3)
pca.fit(df_scale)
pca_scale = pca.transform(df_scale)
pca_df = pd.DataFrame(pca_scale, columns=['pc1', 'pc2', 'pc3'])
print(pca.explained_variance_ratio_)
# 在3D空间中绘制数据，可以看到DBSCAN存在一些潜在的问题
Scene = dict(xaxis=dict(title='PC1'), yaxis=dict(
    title='PC2'), zaxis=dict(title='PC3'))
trace = go.Scatter3d(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1], z=pca_df.iloc[:, 2],
                     mode='markers', marker=dict(colorscale='Greys', opacity=0.3, size=10, ))
layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=1000, width=1000)
data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.show()
# DBSCAN聚类
# 方法1
# 讨论过的“肘形法”来确定合适的epsilon级别。看起来最佳的值在0.2左右
plt.figure(figsize=(10, 5))
nn = NearestNeighbors(n_neighbors=5).fit(pca_df)
distances, idx = nn.kneighbors(pca_df)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()
# 看看下面的3D图，我们可以看到一个包含了大多数数据点的集群
db = DBSCAN(eps=0.2, min_samples=6).fit(pca_df)
labels = db.labels_  # Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" %
      metrics.silhouette_score(pca_df, labels))
Scene = dict(xaxis=dict(title='PC1'), yaxis=dict(
    title='PC2'), zaxis=dict(title='PC3'))
labels = db.labels_trace = go.Scatter3d(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1], z=pca_df.iloc[:, 2], mode='markers', marker=dict(
    color=labels, colorscale='Viridis', size=10, line=dict(color='gray', width=5)))
layout = go.Layout(scene=Scene, height=1000, width=1000)
data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    title='DBSCAN clusters (53) Derived from PCA', font=dict(size=12,))
fig.show()
# 方法2
# 不使用“肘部方法”和最小值启发式方法，而是使用迭代方法来微调我们的DBSCAN模型。在对数据应用DBSCAN算法时，我们将迭代一系列的epsilon和最小点值
pca_eps_values = np.arange(0.2, 1.5, 0.1)
pca_min_samples = np.arange(2, 5)
pca_dbscan_params = list(product(pca_eps_values, pca_min_samples))
pca_no_of_clusters = []
pca_sil_score = []
pca_epsvalues = []
pca_min_samp = []
for p in pca_dbscan_params:
pca_dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(pca_df)
pca_epsvalues.append(p[0])
pca_min_samp.append(p[1])
pca_no_of_clusters.append(
    len(np.unique(pca_dbscan_cluster.labels_)))
pca_sil_score.append(silhouette_score(pca_df, pca_dbscan_cluster.labels_))
pca_eps_min = list(zip(pca_no_of_clusters, pca_sil_score,
                   pca_epsvalues, pca_min_samp))
pca_eps_min_df = pd.DataFrame(pca_eps_min, columns=[
                              'no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
pca_ep_min_df
# 根据Sklearn文档，标签“-1”等同于一个“嘈杂的”数据点，它还没有被聚集到6个高密度的集群中
db = DBSCAN(eps=1.0, min_samples=4).fit(pca_df)
labels = db.labels_  # Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % silhouette_score(pca_df, labels))
# 从6个DBSCAN派生集群的3D图中可以看出，尽管密度较小
Scene = dict(xaxis=dict(title='PC1'), yaxis=dict(title='PC2'), zaxis=dict(
    title='PC3'))  # model.labels_ is nothing but the predicted clusters i.e y_clusters
labels = db.labels_trace = go.Scatter3d(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1], z=pca_df.iloc[:, 2], mode='markers', marker=dict(
    color=labels, colorscale='Viridis', size=10, line=dict(color='gray', width=5)))
layout = go.Layout(scene=Scene, height=1000, width=1000)
data = [trace]
fig = go.Figure(data=data, layout=layout)fig.update_layout(title="'DBSCAN Clusters (6) Derived from PCA'", font=dict(size=12,))
fig.show()

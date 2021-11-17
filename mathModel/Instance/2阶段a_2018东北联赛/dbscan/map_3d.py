
from numpy.lib.shape_base import column_stack
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly


beer = pd.read_excel(
    'D:/program/pythonPractice/mathModel/Instance/2阶段a_2018东北联赛/二阶段/color_map.xls')
# 标准化
X = beer[["Flow", "Press", "cluster_db"]]
pca_df = pd.DataFrame(X)
# 在3D空间中绘制数据，可以看到DBSCAN存在一些潜在的问题
Scene = dict(xaxis=dict(title='Flow'), yaxis=dict(
    title='Pressure'), zaxis=dict(title='Classification'))
trace = go.Scatter3d(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1], z=pca_df.iloc[:, 2],
                     mode='markers', marker=dict(opacity=0.1, size=beer["Speed"], color=beer["name"]))
# colorscale='Greys',
layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=1000, width=1000)
data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.show()
plotly.offline.plot(fig)

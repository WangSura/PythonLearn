# encoding: UTF-8
import pandas as pd
import pygraphviz as pgv

point_location = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/2016b/2016b/铁路网上火车站点名称表.xlsx", index_col="火车站点名称")
link_matrix_2 = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/2016b/2016b/边邻接矩阵1.xlsx", index_col=0)


# 绘图
G = pgv.AGraph(directed=True, concentrate=True)

# 添加节点
for point in point_location.index:
    color = "#5bc49f" if point.startswith("D") else "blue" if point.startswith(
        "F") else "red" if point.startswith("J") else "#000000"

    G.add_node(point, shape="egg", fontsize=50, fontcolor=color, width=1, height=1,
               pos=f"{0.1 * point_location.at[point, 'x坐标']},{0.1 * (point_location.at[point, 'y坐标'])}!")
# 添加边
for start in point_location.index:
    for end in point_location.index:
        if (link_matrix_2.at[start, end]):
            G.add_edge(
                start, end, weight='2', color="blue", penwidth=2)


# 导出图形
G.layout()
G.draw("D:/program/pythonPractice/mathModel/Instance/2016b/2016b/地图.png")

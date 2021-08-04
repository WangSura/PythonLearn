# encoding: UTF-8
import pandas as pd
import pygraphviz as pgv

point_location = pd.read_excel(
    "D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/相关的要素名称及位置坐标数据.xls", index_col="要素编号")
link_matrix_1 = pd.read_excel(
    "D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/边邻接矩阵1.xlsx", index_col=0)
link_matrix_2 = pd.read_excel(
    "D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/边邻接矩阵2.xlsx", index_col=0)


# 绘图
G = pgv.AGraph(directed=False, concentrate=True)

# 添加节点
for point in point_location.index:
    color = "#5bc49f" if point.startswith("D") else "blue" if point.startswith(
        "F") else "red" if point.startswith("J") else "#000000"

    G.add_node(point, shape="none", fontcolor=color, fixedsize=True, width=0.3, height=0.3,
               pos=f"{0.06 * point_location.at[point, 'X坐标（单位：km）']},{0.06 * (point_location.at[point, 'Y坐标（单位：km）'])}!")
G.layout()
G.draw("D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/地图node.png")
# 添加边
for start in point_location.index:
    for end in point_location.index:
        # 主道
        if link_matrix_2.at[start, end]:
            G.add_edge(start, end, color="red", penwidth=2)
        # 支道
        # if ((link_matrix_1.at[start, end]) ^ (link_matrix_2.at[start, end])):
            #G.add_edge(start, end)
            # 支道
        if ((link_matrix_1.at[start, end]) == 1 and (link_matrix_2.at[start, end]) == 0):
            G.add_edge(start, end)
            # 支道
        if ((link_matrix_1.at[start, end]) == 0 and (link_matrix_2.at[start, end]) == 1):
            G.add_edge(start, end)


# 导出图形
G.layout()
G.draw("D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/地图.png")

# encoding: UTF-8
import pandas as pd

point_name = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/2016b/2016b/铁路网上火车站点名称表.xlsx", index_col="火车站点名称").index

link_matrix = pd.DataFrame(0, index=point_name, columns=point_name)

for start_index, start in enumerate(point_name):
    for end_index, end in enumerate(point_name[start_index + 1:]):
        a = input(f"L({start}-{end}) = ")
        if a != '0':
            link_matrix.at[start, end] = link_matrix.at[end, start] = int(a)

print(link_matrix)
link_matrix.to_excel(
    "D:/program/pythonPractice/mathModel/Instance/2016b/2016b/边邻接矩阵1.xlsx")

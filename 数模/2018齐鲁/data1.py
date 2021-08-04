# encoding: UTF-8
import pandas as pd

point_name = pd.read_excel(
    "D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/相关的要素名称及位置坐标数据.xls", index_col='要素编号').index
link_matrix = pd.DataFrame(0, index=point_name, columns=point_name)
print(link_matrix)
for start_index, start in enumerate(point_name):
    for end_index, end in enumerate(point_name[start_index + 1:]):
        if input(f"L({start}-{end}) = ") == "1":
            link_matrix.at[start, end] = link_matrix.at[end, start] = 1

print(link_matrix)
link_matrix.to_excel("D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/边邻接矩阵1.xlsx")

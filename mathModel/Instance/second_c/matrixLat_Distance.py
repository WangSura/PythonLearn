from math import radians, cos, sin, asin, sqrt
import xlrd

import numpy as np


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


def create_data(path1):
    table1 = xlrd.open_workbook(path1).sheets()[0]  # 获取第一个sheet表
    row1 = table1.nrows  # 行数
    col1 = table1.ncols  # 列数
    points = np.zeros((row1, col1))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col1):
        try:
            cols = np.matrix(table1.col_values(x))  # 把list转换为矩阵进行矩阵操作
            points[:, x] = cols  # 按列把数据存进矩阵中
        except:
            print(x)

    # print(datamatrix.shape)
    return points


path1 = "E:/86191/Documents/qq/MobileFile/2021第十四届“认证杯”数学建模网络挑战赛赛题/C1/test.xlsx"
points = create_data(path1)
# print(points)
list1 = []
list2 = []
for i in points:
    for j in points:
        lon1, lat1, lon2, lat2 = i[0], i[1], j[0], j[1]
        list1.append(haversine(lon1, lat1, lon2, lat2))
    list2.append(list1)
    list1 = []
print(list2)
output = open('E:/86191/Documents/qq/MobileFile/2021第十四届“认证杯”数学建模网络挑战赛赛题/C1/make.xlsx', 'w', encoding='gbk')
for i in range(len(list2)):
    for j in range(len(list2[i])):
        output.write(str(list2[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
        output.write('\t')  # 相当于Tab一下，换一个单元格
    output.write('\n')  # 写完一行立马换行
output.close()
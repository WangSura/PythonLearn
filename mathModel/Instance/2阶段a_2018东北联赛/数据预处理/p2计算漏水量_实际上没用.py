import openpyxl
import xlrd
from xlrd import xldate_as_tuple


def find_leak():
    data = xlrd.open_workbook("2018年东北三省数学建模联赛F题附件.xls")  # 打开excel
    table = data.sheet_by_name("朗贝20140415-0612")  # 读sheet
    # 漏水量，字典类型，键为日期，值为当天的漏水量
    leak1 = {}
    leak2 = {}
    # 当天2-5点的最小流量作为漏水量
    minFlow1 = 999
    minFlow2 = 999
    for i in range(1, table.nrows):
        row = table.row_values(i)  # 行的数据放在数组里
        date = xldate_as_tuple(row[0], 0)
        if i != 1 and date[2] != oldDate[2]:  # 比较日期变化
            key = str(oldDate[1])+"-"+str(oldDate[2])
            leak1[key] = minFlow1
            leak2[key] = minFlow2
            minFlow1 = 999
            minFlow2 = 999
        if date[3] >= 2 and date[3] <= 5:
            minFlow1 = min(minFlow1, row[1])
            minFlow2 = min(minFlow2, row[2])
        oldDate = date  # 记录上一个日期
        if i == table.nrows - 1:  # 单独处理最后一天
            key = str(date[1]) + "-" + str(date[2])
            leak1[key] = minFlow1
            leak2[key] = minFlow2
    return leak1, leak2

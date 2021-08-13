# 1.读取数据与简单预处理
import openpyxl
import pandas as pd
import joblib
import os

dirs = 'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型'
if not os.path.exists(dirs):
    os.makedirs(dirs)

# 读取模型
model = joblib.load(dirs+'/model.pkl')

test = pd.read_excel(
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/待判定的数据12.xlsx')
test = test[test.columns[0:27]]
print('预测结果:\n', model.predict(test))

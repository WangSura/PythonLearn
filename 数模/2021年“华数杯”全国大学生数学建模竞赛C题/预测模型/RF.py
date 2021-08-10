import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_excel(
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/处理方便机器学习1.xlsx')
var = "购买意愿"
df = df[df.columns[0:28]]
#df = df[df[var] > 0]

train_labels = pd.DataFrame(df[var])
train_labels = np.array(df[var])
train_features = df.drop(var, axis=1)  # 删去列
feature_list = list(train_features.columns)
train_features = np.array(train_features)
train_features, test_features, train_labels, test_labels = train_test_split(
    train_features, train_labels, test_size=0.1, random_state=42)
rf = RandomForestClassifier(n_estimators=2000, oob_score=True, n_jobs=-1,
                            random_state=42, max_features='auto', min_samples_leaf=12)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)

show_var = "客户编号"
plt.scatter(df[show_var], train_labels)
plt.plot(np.sort(train_labels),
         predictions[np.argsort(train_labels)], color='r')
plt.show()

print(metrics.accuracy_score(test_labels, predictions))
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2))
                       for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(
    feature_importances, key=lambda x: x[1], reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair))
 for pair in feature_importances]
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation='vertical',
        color='r', edgecolor='k', linewidth=1.2)
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()

# 以下为模型应用
# 创建文件目录
dirs = 'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型'
if not os.path.exists(dirs):
    os.makedirs(dirs)

# 保存模型
joblib.dump(rf, dirs+'/rf.pkl')

rf = joblib.load(dirs+'/rf.pkl')
test = pd.read_excel(
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/加上待判定的数据1.xlsx')
test_var = "购买意愿"
test = test[test.columns[0:27]]

print('预测结果:\n', rf.predict(test))

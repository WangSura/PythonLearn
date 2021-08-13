# 1.读取数据与简单预处理
import numpy as np
from pandas.core.frame import DataFrame
import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_excel(
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/处理方便机器学习3.xlsx')
# df = df.replace({'工资': {'低': 0, '中': 1, '高': 2}})

# 2.提取特征变量和目标变量
X = df.drop(columns='购买意愿')
y = df['购买意愿']

# 3.划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=123)
'''
# 4.模型训练及搭建
model = DecisionTreeClassifier(max_depth=3, random_state=123)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[0:100])

# 通过构造DataFrame进行对比
a = pd.DataFrame()  # 创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
a.head()

# 如果要查看整体的预测准确度，可以采用如下代码：
score = accuracy_score(y_pred, y_test)
print(score)

# 或者用模型自带的score函数查看预测准确度
model.score(X_test, y_test)
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba[0:5])

b = pd.DataFrame(y_pred_proba, columns=['不购买概率', '购买概率'])
b.head()
y_pred_proba[:, 1]
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:, 1])
plt.plot(fpr, tpr)
plt.show()
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print(score)
model.feature_importances_

# 通过DataFrame进行展示，并根据重要性进行倒序排列
features = X.columns  # 获取特征名称
importances = model.feature_importances_  # 获取特征重要性

# 通过二维表格形式显示
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
acc = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
acc

acc.mean()
'''
# 指定决策树分类器中各个参数的范围
parameters = {'max_depth': [2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 20], 'criterion': [
    'gini', 'entropy'], 'splitter': ['best', 'random'], 'min_samples_split': [2, 3, 4, 5, 7, 9, 11, 13, 15]}
# 构建决策树分类器
model = DecisionTreeClassifier()  # 这里因为要进行参数调优，所以不需要传入固定的参数了

# 网格搜索
grid_search = GridSearchCV(model, parameters, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

# 获得参数的最优值
print('网格法最优参数:\n', grid_search.best_params_)
print('网格法最优参数\n')
# 根据多参数调优的结果来重新搭建模型
model = DecisionTreeClassifier(
    criterion='entropy', max_depth=17, min_samples_split=2, splitter='best')  # 97不要random_state
model.fit(X_train, y_train)

# 查看整体预测准确度
y_pred = model.predict(X_test)
score = accuracy_score(y_pred, y_test)
print('整体预测准确度\n', score)

# 查看新的AUC值
# 预测不违约&违约概率
y_pred_proba = model.predict_proba(X_test)
y_pred_proba[:, 1]  # 如果想单纯的查看违约概率，即查看y_pred_proba的第二列

score = roc_auc_score(y_test, y_pred_proba[:, 1])
print('新的AUC值,预测不违约&违约概率\n', score)

dirs = 'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型'
if not os.path.exists(dirs):
    os.makedirs(dirs)

# 保存模型
joblib.dump(model, dirs+'/model3.pkl')

model = joblib.load(dirs+'/model3.pkl')
test = pd.read_excel(
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/待判定的数据3.xlsx')
test = test[test.columns[0:27]]
print('预测结果:\n', model.predict(test))
'''
test_var = "购买意愿"
test = test[test.columns[0:27]]
y_pre = model.predict(test)
prey = DataFrame(y_pre)
prey.to_excel(
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/加上待判定的数据pre_y1.xlsx')
print('预测结果:\n', model.predict(test))

y_pre = model.predict(test)
prey = DataFrame(y_pre)
train_labels = pd.DataFrame(test[test_var])
train_labels = np.array(test[test_var])
prey.to_excel(
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/加上待判定的数据pre_y1.xlsx')
plt.scatter(test['客户编号'], test[test_var])
plt.plot(np.sort(train_labels), y_pre[np.argsort(train_labels)], color='r')
plt.show()
'''

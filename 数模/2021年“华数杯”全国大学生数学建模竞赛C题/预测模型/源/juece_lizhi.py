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
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/处理方便机器学习1.xlsx')
# df = df.replace({'工资': {'低': 0, '中': 1, '高': 2}})

# 2.提取特征变量和目标变量
X = df.drop(columns='购买意愿')
#X = df.drop(columns='目标客户编号')
y = df['购买意愿']

# 3.划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=123)

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
    criterion='gini', max_depth=15, min_samples_split=2, splitter='best')  # 97不要random_state
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

###
# 4.模型训练及搭建


# 通过构造DataFrame进行对比
a = pd.DataFrame()  # 创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
a.head()


def show_res(y_test, y_predict):
    """
    结果展示
    :param x_test: 测试集目标值的真实值
    :param y_predict: 预测值
    :return: None
    """
    # 1、画布
    plt.figure()
    # 2、绘图折线图
    x = np.arange(0, len(y_test))
    res = np.argsort(y_predict)
    # 以列表推导式的形式来获取x 按照z 排序规则进行排序之后的结果
    y_test = [y_test[i] for i in res]
    y_predict.sort_values(axis=0)  # 排序不能用原变量命名
    print(2, type(y_predict))

    plt.scatter(x, y_test, s=60, c='blue', marker='o', alpha=1)
    plt.plot(x, y_predict, c='purple')
    # plt.plot(x,y_predict)
    # 增加标题
    plt.title('预测与真实值')
    # 坐标轴
    plt.xlabel('x轴')
    plt.ylabel('客户序号')
    # 图例
    plt.legend(['真实值', '预测值'])
    # 3、展示
    plt.show()
    # plt.savefig('D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/acc1.png')


plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
acc = a['预测值']  # 实际值数据

pre = a['实际值']  # 预测值数据
show_res(acc, pre)

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
plt.plot(fpr, tpr, color='orange', marker='o',
         markerfacecolor='red', markersize=1)
plt.show()
# plt.savefig('D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/ROC1.png')

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

importances = list(model.feature_importances_)
train_features = df.drop('购买意愿', axis=1)  # 删去列
feature_list = list(train_features.columns)
feature_importances = [(feature, round(importance, 2))
                       for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(
    feature_importances, key=lambda x: x[1], reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair))
 for pair in feature_importances]
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(importances, x_values,  orientation='vertical',
        color='orange', edgecolor='k', linewidth=1.2)
# Tick labels for x axis
plt.yticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
# 去掉图例 legend=False
# 缩小柱间距 width=0.8
fig, ax = plt.subplots(figsize=(6, 3))
feature_importances = DataFrame(feature_importances)
feature_importances.plot.barh(legend=False, ax=ax, width=0.8)
plt.yticks(x_values, feature_list, rotation='horizontal')
# 逆序显示 y 轴
# ax.invert_yaxis()

# 去除四周的边框 (spine.set_visible(False))
[spine.set_visible(False) for spine in ax.spines.values()]
plt.show()
# plt.savefig('D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/imp1.png')

###


dirs = 'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型'
if not os.path.exists(dirs):
    os.makedirs(dirs)

# 保存模型
joblib.dump(model, dirs+'/model.pkl')

model = joblib.load(dirs+'/model.pkl')
test = pd.read_excel(
    'D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/预测模型/待判定的数据1.xlsx')
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

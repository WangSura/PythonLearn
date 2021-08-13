 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

#准备训练数据和y值
X_train, y_train = ...
#初步定义分类器
rfc = RandomForestClassifier(max_depth=2, random_state=0)
#需要选择的参数名称一起后选值
tuned_parameters = [{'min_samples_leaf':[1,2,3,4], 'n_estimators':[50,100,200]}]
#神器出场,cv设置交叉验证
clf = GridSearchCV(estimator=rfc,param_grid=tuned_parameters, cv=5, n_jobs=1)
#拟合训练集
clf.fit(X_train, y_train)
print('Best parameters:')
print(clf.best_params_)
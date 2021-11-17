import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


def localoutlierfactor(data, predict, k, group_str):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(
        n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
    clf.fit(data)
    # 记录 LOF 离群因子，做相反数处理
    predict['local outlier factor %s' % group_str] = - \
        clf._decision_function(predict.iloc[:, eval(group_str)])
    return predict


def plot_lof(result, method, group_str):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure('local outlier factor %s' % group_str)
    try:
        plt.scatter(result[result > method].index,
                    result[result > method], c='m', s=50,
                    marker='o', alpha=None,
                    label='Outlier', cmap="Oranges")
    except Exception:
        pass
    try:
        plt.scatter(result[result <= method].index,
                    result[result <= method], c='c', s=50,
                    marker='o', alpha=None, label='normal point', cmap="summer")
    except Exception:
        pass
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF Local outlier detection', fontsize=13)
    plt.ylabel('Local outlier', fontsize=15)
    plt.legend()
    plt.show()


def ensemble_lof(data, predict=None, k=5, groups=[], method=1, vote_method='auto'):
    import pandas as pd
    import numpy as np
    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    data = pd.DataFrame(data)
    predict = pd.DataFrame(predict)
    # 数据标签分组，默认独立自成一组
    for i in range(data.shape[1]):
        if i not in pd.DataFrame(groups).values:
            groups += [[i]]
    # 扩充阈值列表
    if type(method) != list:
        method = [method]
        method += [1] * (len(groups) - 1)
    else:
        method += [1] * (len(groups) - len(method))
    vote = np.zeros(len(predict))
    # 计算LOF离群因子并根据阈值进行票数统计
    for i in range(len(groups)):
        predict = localoutlierfactor(pd.DataFrame(
            data).iloc[:, groups[i]], predict, k, str(groups[i]))
        #plot_lof(predict.iloc[:, -1], method[i], str(groups[i]))
        vote += predict.iloc[:, -1] > method[i]
    # 根据票数阈值划分离群点与正常点
    predict['vote'] = vote
    if vote_method == 'auto':
        vote_method = len(groups)/2
    outliers = predict[vote > vote_method]
    inliers = predict[vote <= vote_method]
    return outliers, inliers


posi = pd.read_excel(
    r'D:/program/pythonPractice/mathModel/Instance/2阶段a_2018东北联赛/二阶段/p3_version1/A题原数据分类后.xls')
lat = np.array(posi["Flow"][:])  # 纬度
lon = np.array(posi["P"][:])
A = list(zip(lat, lon))  # 按照纬度-经度匹配

# 获取会员对任务密度，取第5邻域，阈值为 100，权重分别为5，1
outliers4, inliers4 = ensemble_lof(
    A, predict=None, k=50, groups=[[0, 1]], method=0.12, vote_method='auto')
outliers4.to_excel(
    "D:/program/pythonPractice/mathModel/Instance/2阶段a_2018东北联赛/二阶段/p3_version1/out2.xlsx")
inliers4.to_excel(
    "D:/program/pythonPractice/mathModel/Instance/2阶段a_2018东北联赛/二阶段/p3_version1/in2.xlsx")

# 获取会员对任务密度，取第5邻域，阈值分别为 1.5，2，得票数超过 1 的认为是异常点
'''
# 绘图程序
#plt.scatter(np.array(B)[:, 0], np.array(B)[:, 1],s=20, c='c', alpha=0.5)
plt.scatter(np.array(A)[:, 0], np.array(A)[:, 1], s=20, c='c', alpha=0.8)
plt.scatter(outliers4[0], outliers4[1], s=300 +
            outliers4['local outlier factor [0, 1]']*5000, c=outliers4[1]*10, alpha=0.3, cmap='rainbow')
plt.title('k = 50, method = 0.12')
# title自己要改
plt.show()
# plt.savefig("D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/数据清洗/离散值处理/品牌一离散图.png")
'''

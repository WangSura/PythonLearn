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
                    result[result > method], c='red', s=50,
                    marker='.', alpha=None,
                    label='离群点')
    except Exception:
        pass
    try:
        plt.scatter(result[result <= method].index,
                    result[result <= method], c='black', s=50,
                    marker='.', alpha=None, label='正常点')
    except Exception:
        pass
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF局部离群点检测', fontsize=13)
    plt.ylabel('局部离群因子', fontsize=15)
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
        plot_lof(predict.iloc[:, -1], method[i], str(groups[i]))
        vote += predict.iloc[:, -1] > method[i]
    # 根据票数阈值划分离群点与正常点
    predict['vote'] = vote
    if vote_method == 'auto':
        vote_method = len(groups)/2
    outliers = predict[vote > vote_method].sort_values(by='vote')
    inliers = predict[vote <= vote_method].sort_values(by='vote')
    return outliers, inliers
    if vote_method == 'auto':
        vote_method = len(groups)/2
    outliers = predict[vote > vote_method].sort_values(by='vote')
    inliers = predict[vote <= vote_method].sort_values(by='vote')
    return outliers, inliers


posi = pd.read_excel(
    r'D:\program\pythonPractice\数模\2021年“华数杯”全国大学生数学建模竞赛C题\数据清洗\离散值处理\处理方便机器学习1.xlsx')
lat = np.array(posi["a1"][:])  # 纬度
A = list(zip(lat, np.zeros_like(lat)))  # 按照纬度-经度匹配

posi = pd.read_excel(
    r'D:\program\pythonPractice\数模\2021年“华数杯”全国大学生数学建模竞赛C题\数据清洗\离散值处理\处理方便机器学习1.xlsx')
lon = np.array(posi["a2"][:])  # 经度
B = list(zip(lon, np.zeros_like(lon)))  # 按照纬度-经度匹配

# 获取会员对任务密度，取第5邻域，阈值为 100，权重分别为5，1
outliers4, inliers4 = ensemble_lof(A, B, k=25, method=[0.5, 1], vote_method=1)
outliers4.to_excel(
    "D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/数据清洗/离散值处理/out1.xlsx")
inliers4.to_excel(
    "D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/数据清洗/离散值处理/in1.xlsx")

# 获取会员对任务密度，取第5邻域，阈值分别为 1.5，2，得票数超过 1 的认为是异常点


# 绘图程序
plt.figure('投票集成 LOF 模式')
plt.scatter(np.array(B)[:, 0], np.array(B)[:, 1], s=10, c='b', alpha=0.5)
plt.scatter(np.array(A)[:, 0], np.array(A)[:, 1], s=10, c='green', alpha=0.3)
plt.scatter(outliers4[0], outliers4[1], s=10 + 1000, c='r', alpha=0.2)
plt.title('k = 25, method = [0.5, 1]')
plt.savefig(
    "D:/program/pythonPractice/数模/2021年“华数杯”全国大学生数学建模竞赛C题/数据清洗/离散值处理/品牌一离散图.png")

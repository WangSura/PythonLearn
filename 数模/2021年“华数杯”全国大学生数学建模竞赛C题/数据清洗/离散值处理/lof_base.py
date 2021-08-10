import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


def localoutlierfactor(data, predict, k):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(
        n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
    clf.fit(data)
    # 记录 k 邻域距离
    predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
    # 记录 LOF 离群因子，做相反数处理
    predict['local outlier factor'] = - \
        clf._decision_function(predict.iloc[:, :-1])
    return predict


def plot_lof(result, method):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(8, 4)).add_subplot(111)
    plt.scatter(result[result['local outlier factor'] > method].index,
                result[result['local outlier factor'] > method]['local outlier factor'], c='red', s=50,
                marker='.', alpha=None,
                label='离群点')
    plt.scatter(result[result['local outlier factor'] <= method].index,
                result[result['local outlier factor'] <= method]['local outlier factor'], c='black', s=50,
                marker='.', alpha=None, label='正常点')
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF局部离群点检测', fontsize=13)
    plt.ylabel('局部离群因子', fontsize=15)
    plt.legend()
    plt.show()


def lof(data, predict=None, k=5, method=1, plot=False):
    import pandas as pd
    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
    # 计算 LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    if plot == True:
        plot_lof(predict, method)
    # 根据阈值划分离群点与正常点
    outliers = predict[predict['local outlier factor']
                       > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <=
                      method].sort_values(by='local outlier factor')
    return outliers, inliers


def box(data, legend=True):
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.style.use("ggplot")
    plt.figure()
    # 如果不是DataFrame格式，先进行转化
    if type(data) != pd.core.frame.DataFrame:
        data = pd.DataFrame(data)
    p = data.boxplot(return_type='dict')
    warming = pd.DataFrame()
    y = p['fliers'][0].get_ydata()
    y.sort()
    for i in range(len(y)):
        if legend == True:
            plt.text(1, y[i] - 1, y[i], fontsize=10, color='black', ha='right')
        if y[i] < data.mean()[0]:
            form = '低'
        else:
            form = '高'
        warming = warming.append(
            pd.Series([y[i], '偏' + form]).T, ignore_index=True)
    print(warming)
    plt.show()

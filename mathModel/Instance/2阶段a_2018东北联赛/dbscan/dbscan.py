import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics


def loadData(filePath):
    f = open(filePath)
    lines = f.readlines()
    # print(lines)
    mac2id = dict()
    online_times = []
    for line in lines:
        # lines[1]
        # 2c929293466b97a6014754607e457d68,U201215025,A417314EEA7B,10.12.49.26,2014-07-20 22:44:18.540000000,2014-07-20 23:10:16.540000000,1558,15,本科生动态IP模版,100元每半年,internet
        # .split(',' )[0]   .split(',' )[1]  .......
        mac = line.split(',')[2]
        # 1558时间单位为秒
        online_time = int(line.split(',')[6])
        # line.split(',')[4]  2014-07-20 22:44:18.540000000
        # .split(' ')[1]      22:44:18.540000000
        # .split(':')[0]      22
        start_time = int(line.split(',')[4].split(' ')[1].split(':')[0])
        # print(mac,online_time,start_time)
        if mac not in mac2id:
            mac2id[mac] = len(online_times)
            # print(mac2id)                   #{'A417314EEA7B': 0, 'F0DEF1C78366': 1, '88539523E88D': 2,,,,}
            # print(online_times)            #[(22, 1558), (12, 40261),,,,()]
            online_times.append((start_time, online_time/12000))
        else:
            # 如果有相同的MAC地址 则以最后一条为准 实际上没有
            online_times[mac2id[mac]] = [(start_time, online_time)]
            print(online_times)

    # print(online_times)   [(22, 1558), (12, 40261),,,,,
    # print(np.array(online_times))   .reshape((-1,2))要两列数据 -1为unspecified value
    #  [[    22   1558]
    #  [    12  40261]
    #  [    22   1721].....]
    #
    real_X = np.array(online_times).reshape((-1, 2))
    return real_X


X = loadData("E:\Desktop\python_code\sklearn\课程数据\聚类\\time2.txt")
# print(X)
db = DBSCAN(eps=0.5, min_samples=20, metric='euclidean').fit(X)
labels = db.labels_
print('Labels:', labels)

raito = len(labels[labels[:] == -1]) / len(labels)
print('Noise raito: ', format(raito, '.2%'))
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(' Est imated number of clusters: %d' % n_clusters_)
print(" Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
for i in range(n_clusters_):
    print('Cluster', i, ':')
    print(list(X[labels == i, 0].flatten()))

plt.scatter(X[:, 0], X[:, 1], c=labels)

plt.show()
# plt.hist(X[:,0],24)
# plt.show()

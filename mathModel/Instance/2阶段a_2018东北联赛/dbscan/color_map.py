import matplotlib.pyplot as plt
#import mlp_toolkits.mplot3d as p3d
import pandas as pd
import numpy as np
#from dbscan_check import labels
from matplotlib.colors import Colormap
from sklearn import metrics
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import numpy as np
#from matplotlib import cm
import matplotlib.cm as cm
# 导入数据
beer = pd.read_excel(
    'D:/program/pythonPractice/mathModel/Instance/2阶段a_2018东北联赛/二阶段/color_map.xls')
X = beer[["cluster_db"]]
x = beer[["Press"]]
y = beer[["Flow"]]

# x = np.random.rand(5664) * 140  # 随机产生10个0~2之间的x坐标
plt.figure(num=2)
plt.scatter(x, y, c=X.values, cmap='rainbow')
plt.colorbar()
plt.show()

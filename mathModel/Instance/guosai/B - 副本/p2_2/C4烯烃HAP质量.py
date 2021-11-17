# -*- coding: utf-8 -*-
from matplotlib import colors, patches
from scipy.spatial import ConvexHull
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(context='notebook', style='ticks', rc=rc)

df = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/guosai/B/p2_2/因子.xlsx", sheet_name="C4烯烃HAP质量")
plt.figure(dpi=100,  num="C4烯烃HAP质量")
sns.scatterplot(data=df, x='HAP质量', y='C4烯烃选择性(%)',  hue='控制变量',
                palette="hls", legend=0)
plt.title('仅HAP质量变化')
plt.show()

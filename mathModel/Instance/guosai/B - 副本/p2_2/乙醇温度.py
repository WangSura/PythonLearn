# -*- coding: utf-8 -*-
from matplotlib import patches
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
    "D:/program/pythonPractice/mathModel/Instance/guosai/B/p2_2/因子.xlsx", sheet_name="乙醇温度")
plt.figure(dpi=100, figsize=(6, 6), num="乙醇温度")
sns.lineplot(data=df, x='温度', y='乙醇转化率(%)',  hue='控制变量',
             palette="Set2", sizes=1000, legend=0)
plt.title('仅温度变化')
plt.show()

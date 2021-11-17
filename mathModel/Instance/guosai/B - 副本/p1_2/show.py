from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/guosai/B/p1_2/附件2.xlsx")
plt.figure(dpi=100, figsize=(6, 6), num="B7")
sns.scatterplot(data=df, x='temperature', y='dependent variable value', s=200, hue='dependent variable',
                palette="husl")
fig, axes = plt.subplots(3, 2)  # fig是整个画布，axes是子图,1，2表示1行两列
fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(3, 3)
f3_ax1 = fig3.add_subplot(gs[0, :])
f3_ax1.set_title('gs[0, :]')
ax1 = fig.add_subplot(gs[0, 0:1])
plt.plot([1, 2, 3])
# gs[0, 0:3]中0选取figure的第一行，0:3选取figure第二列和第三列
ax2 = fig.add_subplot(gs[0, 1:3])
f3_ax2 = fig3.add_subplot(gs[1, :-1])
f3_ax2.set_title('gs[1, :-1]')
f3_ax3 = fig3.add_subplot(gs[1:, -1])
f3_ax3.set_title('gs[1:, -1]')
f3_ax4 = fig3.add_subplot(gs[-1, 0])
f3_ax4.set_title('gs[-1, 0]')
f3_ax5 = fig3.add_subplot(gs[-1, -2])
f3_ax5.set_title('gs[-1, -2]')
gs = gridspec
sns.scatterplot(x=struct.标签, y=struct.续作存在百分比, data=struct, ax=axes[0][0])
plt.subplots_adjust(wspace=0.5)  # 子图很有可能左右靠的很近，调整一下左右距离
sns.scatterplot(x=struct.标签, y=struct.续作存在个数, data=struct, ax=axes[1])
fig.set_figwidth(10)  # 这个是设置整个图（画布）的大小
plt.title('The attachment 2')
plt.show()

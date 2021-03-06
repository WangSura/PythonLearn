from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Import Data
df = pd.read_csv(
    "D:/program/pythonPractice/mathModel/Picture/1/datasets-master/mpg_ggplot2.csv")
df_select = df.loc[df.cyl.isin([4, 8]), :]

# Each line in its own column
sns.set_style("white")
gridobj = sns.lmplot(x="displ", y="hwy",
                     data=df_select,
                     height=7,
                     robust=True,
                     palette='Set1',
                     col="cyl",
                     scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))

# Decorations
gridobj.set(xlim=(0.5, 7.5), ylim=(0, 50))
plt.show()

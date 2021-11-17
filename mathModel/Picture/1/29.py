import matplotlib.lines as mlines
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(
    "D:/program/pythonPractice/mathModel/Picture/1/datasets-master/mpg_ggplot2.csv")
# Import Data


# Draw Plot
plt.figure(figsize=(13, 10), dpi=80)
sns.boxplot(x='class', y='hwy', data=df, notch=False)

# Add N Obs inside boxplot (optional)


def add_n_obs(df, group_col, y):
    medians_dict = {grp[0]: grp[1][y].median()
                    for grp in df.groupby(group_col)}
    xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
    n_obs = df.groupby(group_col)[y].size().values
    for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
        plt.text(x, medians_dict[xticklabel]*1.01, "#obs : "+str(n_ob),
                 horizontalalignment='center', fontdict={'size': 14}, color='white')


add_n_obs(df, group_col='class', y='hwy')

# Decoration
plt.title('Box Plot of Highway Mileage by Vehicle Class', fontsize=22)
plt.ylim(10, 40)
plt.show()

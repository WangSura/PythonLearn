import matplotlib.patches as patches
from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Import Data

# Prepare Data
df_raw = pd.read_csv(
    "D:/program/pythonPractice/mathModel/Picture/1/datasets-master/mpg_ggplot2.csv")
# Prepare Data
df = df_raw[['cty', 'manufacturer']].groupby(
    'manufacturer').apply(lambda x: x.mean())
df.sort_values('cty', inplace=True)
df.reset_index(inplace=True)

# Draw plot
fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
ax.hlines(y=df.index, xmin=11, xmax=26, color='gray',
          alpha=0.7, linewidth=1, linestyles='dashdot')
ax.scatter(y=df.index, x=df.cty, s=75, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('Dot Plot for Highway Mileage', fontdict={'size': 22})
ax.set_xlabel('Miles Per Gallon')
ax.set_yticks(df.index)
ax.set_yticklabels(df.manufacturer.str.title(), fontdict={
                   'horizontalalignment': 'right'})
ax.set_xlim(10, 27)
plt.show()

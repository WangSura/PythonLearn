from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/guosai/B/p1可视化/附件1.xlsx", sheet_name="A (17)")
plt.figure(dpi=100, figsize=(6, 6), num="B3")
sns.scatterplot(data=df, x='temperature', y='dependent variable value', s=200, hue='dependent variable',
                palette="husl", alpha=0.5)
plt.title('B3')
plt.show()

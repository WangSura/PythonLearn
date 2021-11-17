
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
warnings.simplefilter('ignore')
sns.set_style("white")
# Import Data
df = pd.read_csv(
    "D:/program/pythonPractice/mathModel/Picture/1/datasets-master/mpg_ggplot2.csv")

df_counts = df.groupby(['hwy', 'cty']).size().reset_index(name='counts')
# Draw Stripplot
fig, ax = plt.subplots(figsize=(16, 10), dpi=80)

sns.stripplot(df_counts.cty, df_counts.hwy,
              sizes=df_counts.counts*35, ax=ax)

# Decorations
plt.title(
    'Counts Plot - Size of circle is bigger as more points overlap', fontsize=22)
plt.show()

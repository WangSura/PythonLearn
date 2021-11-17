# Import Dataset
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Import Data
warnings.simplefilter('ignore')
sns.set_style("white")
# Import Data
df = pd.read_csv(
    "D:/program/pythonPractice/mathModel/Picture/1/datasets-master/mtcars.csv")


# Plot
plt.figure(figsize=(12, 10), dpi=80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns,
            yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

# Decorations
plt.title('Correlogram of mtcars', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Load Dataset
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Import Data
warnings.simplefilter('ignore')
sns.set_style("white")
# Import Data
df = sns.load_dataset('iris')

# Plot
plt.figure(figsize=(10, 8), dpi=80)
sns.pairplot(df, kind="scatter", hue="species", plot_kws=dict(
    s=80, edgecolor="white", linewidth=2.5))
plt.show()

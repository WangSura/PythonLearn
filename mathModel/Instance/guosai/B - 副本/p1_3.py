import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", data=tips)
sns.residplot(x="x", y="y", data=anscombe.query(
    "dataset == 'II'"), scatter_kws={"s": 80})

f, ax = plt.subplots(figsize=(5, 6))
sns.regplot(x="total_bill", y="tip", data=tips, ax=ax)

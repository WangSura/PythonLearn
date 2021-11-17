import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/guosai/B/装料方式/附件1.xlsx", sheet_name="装料方式 (2)")
plt.figure(dpi=100, figsize=(6, 6), num="C4 olefins selectivity")
sns.lineplot(data=df, x='Temperature', y='C4 olefins selectivity', hue='Catalyst combination',
             palette="Set2", sizes=1000)
plt.title('C4 olefins selectivity')
plt.show()

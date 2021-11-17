import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(context='notebook', style='ticks', rc=rc)

df = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/guosai/B/p2_2/因子.xlsx", sheet_name="C4烯烃Co负载量")
plt.figure(dpi=100, figsize=(6, 6), num="C4烯烃Co负载量")
sns.lineplot(data=df, x='Co/SiO2的Co负载量', y='C4烯烃选择性(%)',  hue='控制变量',
             palette="Set2", sizes=1000, legend=0)
plt.title('仅Co负载量变化')
plt.show()

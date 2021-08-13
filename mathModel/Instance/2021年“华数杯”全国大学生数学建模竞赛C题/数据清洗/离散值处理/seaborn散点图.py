
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_style()
tips = pd.read_excel(
    "D:\program\pythonPractice\数模\2021年“华数杯”全国大学生数学建模竞赛C题\散点图1.xlsx")
ax = sns.stripplot(x="用户编号", y="a1", hue="smoker",
                   data=tips, jitter=True, palette="Set2", split=True)
sns.catplot(x="各项满意度", y="得分", data=tips, kind="strip")

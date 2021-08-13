import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns

# 主题风格
# %matplotlib inline  # jutyper编辑语法内嵌画图可省略plt.show
sns.set()  # 默认darkgrid
tips = sns.load_dataset("tips")  # 加载内置数据 餐厅小费
sns.relplot(x="total_bill", y="tip", col="time", hue="smoker",
            style="smoker", size="size", data=tips)

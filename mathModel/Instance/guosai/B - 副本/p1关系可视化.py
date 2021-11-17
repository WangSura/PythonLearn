import seaborn as sns
import matplotlib.pyplot as plt
# 设置风格样式
sns.set(style="ticks", color_codes=True)
# 构建数据
iris = sns.load_dataset("iris")
"""
案例3：
为联合关系绘制散点图，为单变量绘制核密度估计图

字段变量名查看案例a,
由于值为数字的字段变量有4个，故绘制的关系图为4x4

通过指定hue来对数据进行分组(效果通过颜色体现)，
并指定调色板palette来设置不同颜色
"""
sns.pairplot(iris, hue="species", palette="husl")
plt.show()
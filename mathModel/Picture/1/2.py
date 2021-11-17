import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
#import spline

X = np.array([0, 1, 3, 5, 10, 15])  # 设置X轴的点
B = np.array([86, 68, 61, 61, 48, 43])  # Y轴的点   B为研华6333
C = np.array([90, 81, 75, 68, 62, 47])  # C为D-Link
D = np.array([95, 70, 66, 61, 53, 49])  # D为思科1702
E = np.array([98, 76, 67, 61, 51, 43])  # E为Aruba 303
F = np.array([95, 70, 63, 57, 48, 42])  # F为思科1815

# xnew = np.linspace(X.min(),X.max(),300)    #linspace 在x.min和x.max之间取300个点

# plt.scatter(x, y, c='black',alpha = 0.5)  #alpha:透明度) c:颜色

# func1 = interpolate.interp1d(X,B,kind='cubic')   平滑曲线
#ynew1 = func1(xnew)
#func2 = interpolate.interp1d(X,C,kind='cubic')
#ynew2= func2(xnew)
#B_smooth = spline(X,B,xnew)
#C_smooth = spline(X,C,xnew)

# 生成散点图，s代表方形，b代表蓝色blue，-代表直线连接，label代表标签
plt.plot(X, B, 'sb-', label='Advantech 6333')
plt.plot(X, C, 'or-', label=u"D-Link")  # o代表圆形，r代表红色red
plt.plot(X, D, 'py-', label=u"Cisco 1702")  # p代表实心五角星，1代表下花三角，y代表黄色yellow
plt.plot(X, E, 'vg-', label=u"Aruba 303")  # v代表倒三角标记，2代表上花三角，g代表绿色green
# h代表竖六边形，3代表左花三角，m代表洋红色magenta，c代表青色cyan
plt.plot(X, F, 'hm-', label=u"Cisco 1815")

# plt.text(r"R7000")
# plt.plot(xnew,ynew2)
plt.xlabel("Distance(m)")
plt.ylabel("strength(%)")
plt.title("Received Signal Strength Indication")
plt.legend()  # 使得标签生效
plt.show()

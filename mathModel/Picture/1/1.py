import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    pm_15db_3p = [302, 345, 365, 405, 425]
    pm_20db_3p = [312, 356, 375, 417, 439]
    pm_15db_rd = [200, 239, 256, 300, 320]
    pm_20db_rd = [205, 245, 264, 310, 325]
    x = [0.001, 0.005, 0.01, 0.05, 0.099]

    color1 = '#EC4B51'  # red
    color2 = '#26F03C'  # green

    plt.plot(x, pm_15db_3p, '-', label='Pc_max=15dBm,3-partite', color=color1)
    plt.plot(x, pm_20db_3p, '-',  label='Pc_max=20dBm,3-partite', color=color2)
    plt.plot(x, pm_15db_rd, '--', label='Pc_max=15dBm,Randomized', color=color1)
    plt.plot(x, pm_20db_rd, '--', label='Pc_max=20dBm,Randomized', color=color2)

    # 刻度
    x_ticks = np.arange(0, 0.11, 0.01)
    y_ticks = np.arange(150, 450, 50)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # 点的形状
    plt.scatter(x, pm_15db_3p, marker='o', c='gray', edgecolors=color1)
    plt.scatter(x, pm_15db_rd, marker='o', c='gray', edgecolors=color1)

    plt.scatter(x, pm_20db_3p, marker='D', c='gray', edgecolors=color2)
    plt.scatter(x, pm_20db_rd, marker='D', c='gray', edgecolors=color2)

    # 坐标轴显示范围
    plt.axis([0, 0.1, 150, 450])

    # x轴和y轴意义
    plt.xlabel('$p$0')
    plt.ylabel('$\sum_{m=1}$$C_m$(bps/Hz)')

    # 绘制
    plt.legend()
    plt.grid()
    plt.show()

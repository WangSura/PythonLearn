
# -*- coding: utf-8 -*-
"""
aimfunc.py - 目标函数文件
描述:
    目标：max f = 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)
    约束条件：
        x1 != 10
        x2 != 5
        x1 ∈ [-3, 12.1] # 变量范围是写在遗传算法的参数设置里面
        x2 ∈ [4.1, 5.8]
"""

import time
from aimfunc import aimfunc  # 导入自定义的目标函数
import geatpy as ea  # 导入geatpy库
import numpy as np


def aimfunc(Phen, CV):
    x1 = Phen[:, [0]]  # 获取表现型矩阵的第一列，得到所有个体的x1的值
    x2 = Phen[:, [1]]
    f = 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)
    exIdx1 = np.where(x1 == 10)[0]  # 因为约束条件之一是x1不能为10，这里把x1等于10的个体找到
    exIdx2 = np.where(x2 == 5)[0]
    CV[exIdx1] = 1
    CV[exIdx2] = 1
    return [f, CV]


# -*- coding: utf-8 -*-
"""main.py"""

"""============================变量设置============================"""
x1 = [-3, 12.1]  # 第一个决策变量范围
x2 = [4.1, 5.8]  # 第二个决策变量范围
b1 = [1, 1]  # 第一个决策变量边界，1表示包含范围的边界，0表示不包含
b2 = [1, 1]  # 第二个决策变量边界，1表示包含范围的边界，0表示不包含
ranges = np.vstack([x1, x2]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
borders = np.vstack([b1, b2]).T  # 生成自变量的边界矩阵
varTypes = np.array([0, 0])  # 决策变量的类型，0表示连续，1表示离散
"""==========================染色体编码设置========================="""
Encoding = 'BG'  # 'BG'表示采用二进制/格雷编码
codes = [0, 0]  # 决策变量的编码方式，设置两个0表示两个决策变量均使用二进制编码
precisions = [4, 4]  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位
scales = [0, 0]  # 0表示采用算术刻度，1表示采用对数刻度
FieldD = ea.crtfld(Encoding, varTypes, ranges, borders,
                   precisions, codes, scales)  # 调用函数创建译码矩阵
"""=========================遗传算法参数设置========================"""
NIND = 100  # 种群个体数目
MAXGEN = 200  # 最大遗传代数
maxormins = [-1]  # 列表元素为1则表示对应的目标函数是最小化，元素为-1则表示对应的目标函数是最大化
selectStyle = 'rws'  # 采用轮盘赌选择
recStyle = 'xovdp'  # 采用两点交叉
mutStyle = 'mutbin'  # 采用二进制染色体的变异算子
pc = 0.7  # 交叉概率
pm = 1  # 整条染色体的变异概率（每一位的变异概率=pm/染色体长度）
Lind = int(np.sum(FieldD[0, :]))  # 计算染色体长度
obj_trace = np.zeros((MAXGEN, 2))  # 定义目标函数值记录器
var_trace = np.zeros((MAXGEN, Lind))  # 染色体记录器，记录历代最优个体的染色体
"""=========================开始遗传算法进化========================"""
start_time = time.time()  # 开始计时
Chrom = ea.crtpc(Encoding, NIND, FieldD)  # 生成种群染色体矩阵
variable = ea.bs2ri(Chrom, FieldD)  # 对初始种群进行解码
# 初始化一个CV矩阵（此时因为未确定个体是否满足约束条件，因此初始化元素为0，暂认为所有个体是可行解个体）
CV = np.zeros((NIND, 1))
ObjV, CV = aimfunc(variable, CV)  # 计算初始种群个体的目标函数值
FitnV = ea.ranking(maxormins * ObjV, CV)  # 根据目标函数大小分配适应度值
best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
# 开始进化
for gen in range(MAXGEN):
    SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND-1), :]  # 选择
    SelCh = ea.recombin(recStyle, SelCh, pc)  # 重组
    SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)  # 变异
    # 把父代精英个体与子代的染色体进行合并，得到新一代种群
    Chrom = np.vstack([Chrom[best_ind, :].astype(int), SelCh])
    Phen = ea.bs2ri(Chrom, FieldD)  # 对种群进行解码(二进制转十进制)
    ObjV, CV = aimfunc(Phen, CV)  # 求种群个体的目标函数值
    FitnV = ea.ranking(maxormins * ObjV, CV)  # 根据目标函数大小分配适应度值
    # 记录
    best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
    obj_trace[gen, 0] = np.sum(ObjV)/ObjV.shape[0]  # 记录当代种群的目标函数均值
    obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
    var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体
# 进化完成
end_time = time.time()  # 结束计时
ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])  # 绘制图像
"""============================输出结果============================"""
best_gen = np.argmax(obj_trace[:, [1]])
print('最优解的目标函数值：', obj_trace[best_gen, 1])
variable = ea.bs2ri(var_trace[[best_gen], :], FieldD)  # 解码得到表现型（即对应的决策变量值）
print('最优解的决策变量值为：')
for i in range(variable.shape[1]):
    print('x'+str(i)+'=', variable[0, i])
print('用时：', end_time - start_time, '秒')

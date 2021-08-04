import gurobipy
import pandas as pd
 
from data3_shortest import cost_matrix, A_graph, B_graph
 
# %% 1. 数据准备
# 导入文件
A_cost_matrix = pd.read_excel("D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/A车代价矩阵.xlsx", index_col=0)
B_cost_matrix = pd.read_excel("D:/program/pythonPractice/数模/2018齐鲁/暑假第一阶段/B车代价矩阵.xlsx", index_col=0)
 
# 初始化映射表
cost_matrix = {"A": A_cost_matrix, "B": B_cost_matrix}
graph = {"A": A_graph, "B": B_graph}
alpha = {i: "B" if i in [7, 8, 9, 10, 17, 18, 19, 20] else "A" for i in range(1, 21)}
beta = {i: "D1" if i in range(1, 11) else "D2" for i in range(1, 21)}
w = {"A": 20 / 60, "B": 15 / 60}
I = range(1, 21)
J = [point for point in A_cost_matrix.index if point.startswith("F")]
K = [point for point in A_cost_matrix.index if point.startswith("Z")]
 
# %% 2. Gurobi 求解
MODEL = gurobipy.Model()
 
# 创建变量
x = MODEL.addVars(I, J, K, J, K, vtype=gurobipy.GRB.BINARY)
t = MODEL.addVars(I)
t_max = MODEL.addVar()
 
# 更新变量空间
MODEL.update()
 
# 创建目标函数
MODEL.setObjectiveN(t_max, priority=1, index=0)
MODEL.setObjectiveN(t.sum(), priority=0, index=1)
 
# 创建约束条件
MODEL.addConstrs(sum(x[i, j1, k1, j2, k2] for j1 in J for k1 in K for j2 in J for k2 in K) == 1 for i in I)
MODEL.addConstrs(sum(x[i, j, k1, j_, k2] + x[i, j_, k1, j, k2] for i in I for j_ in J for k1 in K for k2 in K) <= 1 for j in J)
MODEL.addConstrs(sum(x[i, j1, k_, j2, k] + x[i, j1, k, j2, k_] for i in I for j1 in J for k_ in K for j2 in J) <= 8 for k in K)
MODEL.addConstrs(t[i] == sum((cost_matrix[alpha[i]].at[beta[i], j1] + cost_matrix[alpha[i]].at[j1, k1] + cost_matrix[alpha[i]].at[k1, j2] + cost_matrix[alpha[i]].at[j2, k2]) * x[i, j1, k1, j2, k2] for j1 in J for k1 in K for j2 in J for k2 in K) + 2 * w[alpha[i]] for i in I)
MODEL.addConstrs(t_max >= t[i] for i in I)
 
# 执行最优化
MODEL.optimize()
 
# 输出结果
print(f"任务完成用时：{round(t_max.x, 2)} h")
print(f"平均用时：{round(t.sum().getValue() / 20, 2)} h")
 
def fun(i):
    for j1 in J:
        for k1 in K:
            for j2 in J:
                for k2 in K:
                    if x[i, j1, k1, j2, k2].x:
                        path1 = graph[alpha[i]].shortest_paths(beta[i], j1, show=False)[0][0][:-1]
                        path2 = graph[alpha[i]].shortest_paths(j1, k1, show=False)[0][0][:-1]
                        path2[0] = path2[0] + "(作业点 1)"
                        path3 = graph[alpha[i]].shortest_paths(k1, j2, show=False)[0][0][:-1]
                        path3[0] = path3[0] + "(补水点 1)"
                        path4 = graph[alpha[i]].shortest_paths(j2, k2, show=False)[0][0]
                        path4[0] = path4[0] + "(作业点 2)"
                        path4[-1] = path4[-1] + "(补水点 2)"
                        points = path1 + path2 + path3 + path4
                        print(f"编号：{alpha[i]}-{i}t用时：{round(t[i].x, 2)}ht路线：{' -> '.join(points)}")
                        return
 
for i in I:
    fun(i)
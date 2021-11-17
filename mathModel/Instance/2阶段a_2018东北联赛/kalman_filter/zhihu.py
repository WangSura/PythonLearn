# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:46:37 2021

@author: xuelin
"""
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

x = np.matrix([[0.0],
               [1.0]])    # 初始状态
x_hat = np.matrix([[0.0],
                   [1.0]])  # 后验估计初始值

Q = np.matrix([[0.5, 0.0],
               [0.0, 0.5]])    # 过程噪声协方差
R = np.matrix([[0.5, 0.0],
               [0.0, 0.5]])    # 测量噪声协方差

A = np.matrix([[1, 1],
               [0, 1]])    # 系统矩阵
H = np.matrix([[1.0, 0.0],
               [0.0, 1.0]])  # 测量矩阵
p = np.matrix([[1.0, 0.0],
               [0.0, 1.0]])  # 后验误差协方差初始值

# 用于记录
x_log = x
z_log = np.matrix([[], []])
x_hat_log = x_hat
x_phat_log = np.matrix([[], []])    # 记录先验估计值

N_step = 30
for i in range(0, N_step):

    # 状态转移
    P_noise = rd.randn(Q.shape[0], 1)*np.diag(Q).reshape(-1, 1)**0.5
    x = A*x + P_noise
    x_log = np.hstack((x_log, x))

    # 测量
    M_noise = rd.randn(R.shape[0], 1)*np.diag(R).reshape(-1, 1)**0.5
    z = H*x + M_noise
    z_log = np.hstack((z_log, z))

    # 预测
    x_phat = A*x_hat    # 先验估计值
    p_p = A*p*A.T + Q   # 先验误差协方差
    x_phat_log = np.hstack((x_phat_log, x_phat))

    # 校正
    k = p_p*H.T*(H*p_p*H.T + R)**-1
    x_hat = x_phat + k*(z - H*x_phat)
    p = (np.mat(np.identity(2)) - k*H)*p_p
    x_hat_log = np.hstack((x_hat_log, x_hat))

legend_text = ['Real', 'Measure', 'Prior', 'Posterior']
plt.figure(figsize=(8, 6))

plt.subplot(211)
plt.plot(range(0, N_step+1), x_log[0, :].T,
         range(1, N_step+1), z_log[0, :].T,
         range(1, N_step+1), x_phat_log[0, :].T,
         range(0, N_step+1), x_hat_log[0, :].T, marker='.')
plt.legend(legend_text)
plt.xlabel('Time($s$)')
plt.ylabel('Position($m$)')

plt.subplot(212)
plt.plot(range(0, N_step+1), x_log[1, :].T,
         range(1, N_step+1), z_log[1, :].T,
         range(1, N_step+1), x_phat_log[1, :].T,
         range(0, N_step+1), x_hat_log[1, :].T, marker='.')
plt.legend(legend_text)
plt.xlabel('Time($s$)')
plt.ylabel('Velocity($m\cdot s^{-1}$)')

plt.savefig('KF.svg')

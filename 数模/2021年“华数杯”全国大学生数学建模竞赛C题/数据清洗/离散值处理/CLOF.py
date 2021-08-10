elbow_method(data, maxtest=11)
elbow_method(B, maxtest=11)
B_cut = kmeans(B, n_clusters=3, m=3)
# 获取会员分布密度，取第10邻域，阈值为3（LOF大于3认为是离群值）
outliers6, inliers6 = lof(B_cut, k=10, method=3)
# 绘图程序
plt.figure('CLOF 离群因子检测')
plt.scatter(np.array(B)[:, 0], np.array(B)[:, 1], s=10, c='b', alpha=0.5)
plt.scatter(outliers6[0], outliers6[1], s=10 +
            outliers6['local outlier factor']*10, c='r', alpha=0.2)
plt.title('k = 10, method = 3')

G = nx.path_graph(5, create_using = nx.DiGraph())  
nx.draw(G,with_labels=True)
plt.title('有向图',fontproperties=myfont)
plt.axis('on')
plt.xticks([])
plt.yticks([])
plt.show()

#计算加权图最短路径长度和前驱
pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)
print('\n加权图最短路径长度和前驱: ',pred, dist)

#返回G中从源到目标的最短加权路径,要求边权重必须为数值
print('\nG中从源0到目标4的最短加权路径: ',nx.dijkstra_path(G,0,4))
print('\nG中从源0到目标4的最短加权路径的长度: ',nx.dijkstra_path_length(G,0,4))  #最短路径长度

#单源节点最短加权路径和长度。
length1, path1 = nx.single_source_dijkstra(G, 0)
print('\n单源节点最短加权路径和长度: ',length1, path1)
#下面两条和是前面的分解
# path2=nx.single_source_dijkstra_path(G,0)
# length2 = nx.single_source_dijkstra_path_length(G, 0)
#print(length1,'$', path1,'$',length2,'$',path2)

#多源节点最短加权路径和长度。
path1 = nx.multi_source_dijkstra_path(G, {0, 4})
length1 = nx.multi_source_dijkstra_path_length(G, {0, 4})

print('\n多源节点最短加权路径和长度:', path1,length1)

#两两节点之间最短加权路径和长度。
path1 = dict(nx.all_pairs_dijkstra_path(G))
length1 = dict(nx.all_pairs_dijkstra_path_length(G))
print('\n两两节点之间最短加权路径和长度: ',path1,length1)

#双向搜索的迪杰斯特拉
length, path = nx.bidirectional_dijkstra(G, 0, 4)
print('\n双向搜索的迪杰斯特拉:',length, path)
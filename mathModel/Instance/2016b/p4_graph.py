from os import name
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.shortest_paths import weighted
from networkx.classes.function import get_edge_attributes, get_node_attributes
import pandas as pd
point_location = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/2016b/2016b/铁路网上火车站点名称表.xlsx", index_col="火车站点名称")
link_matrix_2 = pd.read_excel(
    "D:/program/pythonPractice/mathModel/Instance/2016b/2016b/边邻接矩阵1.xlsx", index_col=0)


def main():
    G = nx.DiGraph()

    # topology construction logic
    for point in point_location.index:
        color = "#5bc49f" if point.startswith("D") else "blue" if point.startswith(
            "F") else "red" if point.startswith("J") else "#000000"

    G.add_node(point, shape="egg", fontsize=50, fontcolor=color, width=1, height=1,
               pos=f"{0.1 * point_location.at[point, 'x坐标']},{0.1 * (point_location.at[point, 'y坐标'])}!")
    for start in point_location.index:
        for end in point_location.index:
            if (link_matrix_2.at[start, end]):
                G.add_edge(
                    start, end, weight=int(link_matrix_2.at[start, end]))
    # draw graph with labels
    pos = nx.random_layout(G)
    # error: pos = f"{0.1 * point_location.at[point, 'x坐标']}, {0.1 * (point_location.at[point, 'y坐标'])}!"
    nx.draw(G, pos)

    # generate node_labels manually

    nx.draw_networkx_labels(
        G, pos)  # pos 又不能省

    # generate edge_labels manually

    nx.draw_networkx_edge_labels(G, pos, get_edge_attributes(G, 'weight'))

    plt.show()


if __name__ == '__main__':
    main()

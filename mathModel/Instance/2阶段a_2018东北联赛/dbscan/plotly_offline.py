# 导入依赖库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import Image
import cufflinks as cf
cf.go_offline()  # 这两句是离线生成图片的设置
cf.set_config_file(offline=True, world_readable=True)
init_notebook_mode(connected=True)
%matplotlib inline


def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])
    positional_encoding[1:, 0::2] = np.sin(
        positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(
        positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    # 归一化, 用位置嵌入的每一行除以它的模长
    # denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    # position_enc = position_enc / (denominator + 1e-8)
    return positional_encoding


positional_encoding = get_positional_encoding(max_seq_len=100, embed_dim=128)

# 3d可视化
relation_matrix = np.dot(positional_encoding, positional_encoding.T)[1:, 1:]
data = [go.Surface(z=relation_matrix)]
layout = go.Layout(scene={"xaxis": {'title': "sequence length"}, "yaxis": {
                   "title": "sequence length"}})
fig = go.Figure(data=data, layout=layout)
iplot(fig)

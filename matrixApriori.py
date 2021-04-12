import pandas as pd
import numpy as np
from itertools import combinations
from operator import itemgetter
from time import time
import warnings
warnings.filterwarnings("ignore")

# 使用apriori包进行分析
from apyori import apriori
dataset = pd.read_csv('retail.csv', usecols=['items'])
def create_dataset(data):
    for index, row in data.iterrows():
        data.loc[index, 'items'] = row['items'].strip()
    data = data['items'].str.split(" ", expand = True)
    # 按照list来存储
    output = []
    for i in range(data.shape[0]):
        output.append([str(data.values[i, j]) for j in range(data.shape[1])])
    return output

dataset = create_dataset(dataset)
association_rules = apriori(dataset, min_support = 0.05, min_confidence = 0.7, min_lift = 1.2, min_length = 2)
association_result = list(association_rules)
association_result
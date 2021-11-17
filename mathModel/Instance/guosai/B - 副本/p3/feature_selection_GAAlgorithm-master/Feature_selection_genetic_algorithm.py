# -*- encoding: utf-8 -*-

import random
import math
import numpy as np
import lightgbm as lgb
import pandas as pd
from Genetic_algorithm import GA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class FeatureSelection(object):
    def __init__(self, aLifeCount=10):
        # self.columns = ['target', 'credit_score', 'overdraft', 'quota', 'quota_is_zero', 'quota_surplus', 'quota_rate', 'credit_score_rank', 'all_is_null_x', 'all_is_zero', 'credit_score_is_null', 'quota_surplus_is_null', 'unit_price_mean', 'unit_price_max', 'unit_price_min', 'unit_price_std', 'record_is_unique', 'auth_id_card_is_null', 'auth_time_is_null', 'phone_is_null', 'all_is_null_y', 'all_not_null', 'card_time_is_null', 'time_phone_is_null', 'feature_1', 'register_days', 'day_mean', 'day_max', 'day_min', 'order_record_count', 'order_record_unique']
        self.columns = ['target', '装料方式', 'Co/SiO2质量',
                        'Co/SiO2的Co负载量', 'HAP质量', '石英砂质量', '乙醇浓度(ml/min)', '温度']
        self.train_data = pd.read_excel(
            r'D:/program/pythonPractice/mathModel/Instance/guosai/B/p3/feature_selection_GAAlgorithm-master/dataSet/因子.xlsx', sheet_name='拆分',  usecols=self.columns)
        # 由于特征数量较多，这里只读取了上面的部分特征 #
        self.validate_data = pd.read_excel(
            r'D:/program/pythonPractice/mathModel/Instance/guosai/B/p3/feature_selection_GAAlgorithm-master/dataSet/因子.xlsx', sheet_name='拆分',  usecols=self.columns)
        self.lifeCount = aLifeCount
        self.ga = GA(aCrossRate=0.7,
                     aMutationRage=0.1,
                     aLifeCount=self.lifeCount,
                     aGeneLenght=len(self.columns) - 1,
                     aMatchFun=self.matchFun())

    def auc_score(self, order):
        print(order)
        features = self.columns[1:]
        features_name = []
        for index in range(len(order)):
            if order[index] == 1:
                features_name.append(features[index])

        labels = np.array(self.train_data['target'], dtype=np.int8)
        d_train = lgb.Dataset(self.train_data[features_name], label=labels)
        params = {
            'boosting': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'train_metric': False,
            'subsample': 0.8,
            'learning_rate': 0.05,
            'num_leaves': 96,
            'num_threads': 4,
            'max_depth': 5,
            'colsample_bytree': 0.8,
            'lambda_l2': 0.01,
            'verbose': -1,     # inhibit print info #
        }
        rounds = 100
        watchlist = [d_train]
        bst = lgb.train(params=params, train_set=d_train,
                        num_boost_round=rounds, valid_sets=watchlist, verbose_eval=10)
        predict = bst.predict(self.validate_data[features_name])
        print(features_name)
        score = roc_auc_score(self.validate_data['target'], predict)
        print('validate score:', score)
        return score

    def matchFun(self):
        return lambda life: self.auc_score(life.gene)

    def run(self, n=0):
        distance_list = []
        generate = [index for index in range(1, n + 1)]
        while n > 0:
            self.ga.next()
            # distance = self.auc_score(self.ga.best.gene)
            distance = self.ga.score
            distance_list.append(distance)
            print(("第%d代 : 当前最好特征组合的线下验证结果为：%f") %
                  (self.ga.generation, distance))
            n -= 1

        print('当前最好特征组合:')
        string = []
        flag = 0
        features = self.columns[1:]
        for index in self.ga.gene:
            if index == 1:
                string.append(features[flag])
            flag += 1
        print(string)
        print('线下最高为auc：', self.ga.score)

        '''画图函数'''
        plt.plot(generate, distance_list)
        plt.xlabel('generation')
        plt.ylabel('distance')
        plt.title('generation--auc-score')
        plt.show()


def main():
    fs = FeatureSelection(aLifeCount=20)
    rounds = 100    # 算法迭代次数 #
    fs.run(rounds)


if __name__ == '__main__':
    main()

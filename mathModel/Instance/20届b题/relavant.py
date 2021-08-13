import csv
import json
import math

filename = "data1.json"
# 处理基础数据 转换成字典
#     key为分区代码  相应的value为嵌套字典
#         键为某一分区内 POI的类型代码 相应的值为该类POI出现的次数


def data_parse(filename):
    number_counts = {}
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for l in data:
            if l["number"] not in number_counts.keys():
                number_counts[l["number"]] = {}
                if l["type"] not in number_counts[l["number"]].keys():
                    number_counts[l["number"]][l["type"]] = 1
            else:
                if l["type"] not in number_counts[l["number"]].keys():
                    number_counts[l["number"]][l["type"]] = 1
                else:
                    number_counts[l["number"]][l["type"]] += 1
    return number_counts

# 计算TF值（TF=当前分区含有的I类POI数目/当前分区含有的POI数目）


def tf(number_counts):
    for m in number_counts:
        fenmu = 0
        for j in number_counts[m]:
            fenmu += number_counts[m][j]
        for k in number_counts[m]:
            number_counts[m][k] = number_counts[m][k]/fenmu
    return number_counts

# 计算IDF的值


def idf(number_counts):
    idf = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0, "g": 0, "h": 0}
    for l in idf:
        count = 0
        D = 0
        for m in number_counts:
            D += 1
            if l in number_counts[m].keys():
                count += 1
        idf[l] = math.log(D/count)
    return idf

# 计算TF-IDF的值


def TFIDF(tf, idf):
    for m in tf:
        for k in idf:
            if k in tf[m].keys():
                tf[m][k] = tf[m][k]*idf[k]
                # for j in tf[m]:
                #     print(m,k)
                #     tf[m][j] = idf[k]*tf[m][j]
                #     print(tf[m][j])
    return tf

# 将最终的TF值写入成json文件


def writetojson(number_counts, filename):
    with open(filename, 'a') as json_file:
        json.dump(number_counts, json_file, indent=2, ensure_ascii=False)


def read_area(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        area = json.load(f)
    return area


def relative(tf_idf, idf, area):
    # N是第i类所占的分区数
    # tl_pd 是（第i类所在分区的tf_idf值）与（第i类所在分区的栅格值）的乘积 后的求和
    # sum_2 是 （第i类所在分区的tf_idf值）的平方的求和
    # sum 是 第 （第i类所在分区的tf_idf值）的求和
    # PD 是第i类所在的分区的栅格值的求和
    # PD_2 是第i类所在的分区的栅格值的平方的求和

    rou = {}
    for m in idf:
        N = 0
        tl_pd = 0
        sum_2 = 0
        sum = 0
        PD = 0
        PD_2 = 0
        for l in tf_idf:
            if m in tf_idf[l].keys():
                N += 1
            if m in tf_idf[l].keys():
                sum_2 += tf_idf[l][m]*tf_idf[l][m]
                tl_pd += tf_idf[l][m]*area[l]
                sum += tf_idf[l][m]
        for l in tf_idf:
            if m in tf_idf[l].keys():
                PD += area[l]
                PD_2 += area[l]*area[l]
        fenzi = N*tl_pd-sum*PD
        fenmu = (N*sum_2-sum*sum)**0.5 - (N*PD_2-PD*PD)**0.5
        poi_md = fenzi/fenmu
        rou[m] = poi_md
    return rou


if __name__ == '__main__':
    filename = "data1.json"
    number_counts = data_parse(filename)
    tf = tf(number_counts)
    idf = idf(number_counts)
    tf_idf = TFIDF(tf, idf)
    print(tf_idf)
    area = read_area('area.json',)
    rou = relative(tf_idf, idf, area)
    rou = sorted(rou.items(), key=lambda x: x[1], reverse=False)
    writetojson(rou, "rou.json")

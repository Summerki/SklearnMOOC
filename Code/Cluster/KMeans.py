# 使用sklearn使用KMeans
# 一篇不错的博客：https://blog.csdn.net/sinat_26917383/article/details/70240628

import numpy as np
from sklearn.cluster import KMeans


# 加载指定数据集名称的数据
def loadData(fileName):
    fr = open('./ClusterDataSet/' + fileName)
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(',')
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retCityName, retData




if __name__ == '__main__':
    cityName, data = loadData('31省市居民家庭消费水平-city.txt')
    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)
    print(label)  # 初始化聚类中心4块，所以输入的data会被分为4类，每类的label在0-3

    # 这里的cluster_centers_会有4行，对应4个类别
    # 每行有8列数据，对应原数据集中的8列消费支出，表示训练后每一类中每一类支出的聚类中心
    print(km.cluster_centers_)  # cluster_centers_聚类中心均值向量矩阵

    expenses = np.sum(km.cluster_centers_, axis=1)  # axis=0在列上相加   axis=1在行上相加
    print(expenses)

    cityCluster = [[], [], [], []]  # 所以这里的4个[]就是为了统计每个类别里的具体信息的，对应上面的4类
    for i in range(len(cityName)):
        cityCluster[label[i]].append(cityName[i])

    print(len(cityCluster))
    for i in range(len(cityCluster)):
        print('Expenses: %.2f'%expenses[i])
        print(cityCluster[i])


# 一种聚类算法
# 专用于弥补KMeans只能聚类出球状的缺点

import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

# 数据介绍：记录编号，学生编号，MAC地址，IP地址，开始上网地址，停止上网地址，上网时长
# 一定要注意，下面是所有样本开始上网的时间点的聚类！！！

mac2id = dict()
onlinetimes = []

f = open('./ClusterDataSet/学生月上网时间分布-TestData.txt', encoding='utf-8')
lines = f.readlines()
for line in lines:
    items = line.strip().split(',')
    mac = items[2]
    onlinetime = int(items[6])
    starttime = int(items[4].split(' ')[1].split(':')[0])  # 意思是取时分秒中的时
    if mac not in mac2id:
        mac2id[mac] = len(onlinetimes)  # mac-->onlinetimes列表的长度
        onlinetimes.append((starttime, onlinetime))
    else:
        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]


real_X = np.array(onlinetimes).reshape((-1, 2))


X = real_X[:, 0:1]
print('X')
print(X)  # 开始上网的时间（时分秒中的时）
print('real_X')
print(real_X)
# eps:两个样本之间的最大距离，即扫描半径
# min_samples ：作为核心点的话邻域(即以其为圆心，eps为半径的圆，含圆上的点)中的最小样本数(包括点本身)
db = skc.DBSCAN(eps=0.01, min_samples=20).fit(X)
labels = db.labels_  # 直接训练完都不用predict就会有label
# labels中存放的对应的簇的标签，如果是-1则代表是noise


print('Labels:')
print(labels)
ratio = len(labels[labels[:] == -1]) / len(labels)   # noise率
print('Noise ratio:', format(ratio, '.2%'))

n_cluster_ = len(set(labels)) - (1 if -1 in labels else 0)  # DBSCAN最后簇的数量

print('Estimated number of cluster: %d'%n_cluster_)
print('Silhouette Coefficient: %0.3f'%metrics.silhouette_score(X, labels))  # 轮廓系数
# 参考：https://blog.csdn.net/sinat_26917383/article/details/70577710

for i in range(n_cluster_):
    print('Cluster ', i, ':')
    print(list(X[labels == i].flatten()))

plt.hist(X, 24)
plt.show()


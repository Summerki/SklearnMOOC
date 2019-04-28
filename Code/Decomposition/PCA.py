# 使用sklearn进行鸢尾花（高维数据集）进行PCA降维可视化
# 鸢尾花数据集包括4个特征：萼片长度，萼片宽度，花瓣长度，花瓣宽度
# 还有1个标签label--》表示哪一类，共被分为3类样本

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
y = data.target  # label
X = data.data  # 特征feature
print(y)
print(X)

pca = PCA(n_components=2)  # 指定PCA降维成2维，即feature保留2维
reduced_X = pca.fit_transform(X)  # fit_transform = fit + transform
# 对原始数据降维，将结果保留在reduced_X中
print(reduced_X)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
    if y[i] == 0:  # label为0的情况
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    if y[i] == 1:  # label为1的情况
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:  # lable为2的情况
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])


# 开始绘图
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
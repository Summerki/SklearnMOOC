# 简单回归示例（只有一个自变量的情况）

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 读取数据集
datasets_X = []
datasets_Y = []
fr = open(r'./DataSet/price.txt', 'r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))


length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length, 1])
datasets_Y = np.array(datasets_Y)

minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX, maxX).reshape([-1, 1])  # X范围：[minX, maxX - 1]
print(minX)
print(maxX)
print(X)


linear = linear_model.LinearRegression()
linear.fit(datasets_X, datasets_Y)


# 画图显示
plt.scatter(datasets_X, datasets_Y, color='red')
plt.plot(X, linear.predict(X), color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
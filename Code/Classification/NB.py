# 测试sklearn中的朴素贝叶斯分类器

import numpy as np
X = np.array([[-1,-1], [-2,-1], [-3,-2], [1,1], [2,1], [3,2]])
y = np.array([1,1,1,2,2,2])

from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯分类器

clf = GaussianNB(priors=None)  # priors为先验概率

clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))  # 输出[1],表示被分到label=1这一类
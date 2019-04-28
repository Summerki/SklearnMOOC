# 使用sklearn实现KNN算法


from sklearn.neighbors import KNeighborsClassifier


# 创建一组数据x和它对应的标签y
x = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

# n_neighbors表示使用最近的几个邻居作为分类的标准
neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(x, y)  # 训练

test = [[1.1], [1.2], [1.3]]
print(neigh.predict(test))   # 预测


# 结果输出[0, 0, 0]
# 表示都预测到了label中的第0类
# 输入样本数据与训练数据之间的距离按从小到大取前K个，称为KNN


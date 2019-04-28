# 使用sklearn实践决策树

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器


from sklearn.model_selection import cross_val_score  # 导入计算交叉验证值的函数


clf = DecisionTreeClassifier()  # 创建一颗基于基尼系数（默认）的决策树
iris = load_iris()

print(cross_val_score(clf, iris.data, iris.target, cv=10))  # cv代表几折交叉验证


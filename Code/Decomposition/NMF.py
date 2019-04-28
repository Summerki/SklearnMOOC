# 非负矩阵分解NMF

from numpy.random import RandomState  # 产生随机种子
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)


# 加载数据集
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
faces = dataset.data



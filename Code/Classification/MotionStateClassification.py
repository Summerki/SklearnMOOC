# 运动状态程序编写

'''
数据集特点：
总共有5个文件夹，代表5名用户的数据
每个用户数据文件夹里面有一个feature文件XXX.feature和一个label文件XXX.label
feature文件包括41列feature，分布如下

1|2|3-15|16-28|29-41
时间戳|心率|传感器1数据|传感器2数据|传感器3数据

label文件一共有0-24共25种身体姿态label
'''


import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer  # 导入预处理模块
from sklearn.model_selection import train_test_split  # 导入自动生成trainSet和testSet的模块
from sklearn.metrics import classification_report  # 导入预测结果评估模块

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# 读取特征文件列表和标签文件中的内容并返回
def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0, 41))
    print(feature)
    label = np.ndarray(shape=(0,1))
    print(label)

    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)  # 指定分隔符为逗号 缺失值为问号 文件中不包含表头行

        #  使用Imputer函数,通过设定strategy参数为'mean'
        #  使用平均值对缺失数据补全
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)

        df = imp.transform(df)

        feature = np.concatenate((feature, df))


    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))

    label = np.ravel(label)
    return feature, label


if __name__ == '__main__':
    featurePaths = [r'./MotionStateDataSet/A/A.feature', r'./MotionStateDataSet/B/B.feature', r'./MotionStateDataSet/C/C.feature', r'./MotionStateDataSet/D/D.feature', r'./MotionStateDataSet/E/E.feature']
    labelPaths = ['MotionStateDataSet/A/A.label', 'MotionStateDataSet/B/B.label', 'MotionStateDataSet/C/C.label', 'MotionStateDataSet/D/D.label', 'MotionStateDataSet/E/E.label']
    # 将A,B,C,D作为trainSet
    x_train, y_train = load_datasets(featurePaths[:4], labelPaths[:4])
    # 将E作为testSet
    x_test, y_test = load_datasets(featurePaths[4:], labelPaths[4:])

    # 使用train_test_split()函数,通过设置测试集比例test_size为0,将数据随机打乱,便于后续分类器的初始化和训练
    # 其实这是一个小技巧，可以将指定数据打乱再返回
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)


    # KNN
    print('Start training KNN')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('KNN training done')
    answer_knn = knn.predict(x_test)
    print('KNN predict done')

    # Decision Tree
    print('Start training Decision Tree')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Decision Tree training done')
    answer_dt = dt.predict(x_test)
    print('Decision Tree predict done')

    # NB
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Bayes Training done')
    answer_gnb = gnb.predict(x_test)
    print('Bayes predict done')

    # 使用classification_report对分类结果衡量
    print('The classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('The classification report for Decison Tree:')
    print(classification_report(y_test, answer_dt))
    print('The classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))
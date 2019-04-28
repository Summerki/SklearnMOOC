# 使用sklearn中的SVM
# 数据集特点：
# 给出当前时间前150天的历史数据预测当天上证指数的涨跌

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

# read_csv:参数一:数据源.encoding:编码格式.parse_dates:第n列解析为日期.index_col:用作索引的列编号
# 将第0列解析为日期，又将第0列直接作为索引列了，也就是将日期列作为索引列
data = pd.read_csv(r'./StockDataSet/000777.csv', encoding='gbk', parse_dates=[0], index_col=0)
# print(data)

# sort_index:参数一:按0列排,ascending(true)升序,inplace:排序后是否覆盖原数据
# 按照日期升序排列
data.sort_index(0, ascending=True, inplace=True)
# print(data)





dayfeature = 150  # 选取150天的数据
featurenum = 5 * dayfeature  # 选取5个特征*150天数
# data.shape[0]-dayfeature:因为我们要用150天数据做训练,对于条目为200条的数据,只有50条数据有前150天的数据来训练的,所以训练集的大小就是200-150
# 对于每一条数据,他的特征是前150天的甩有特征数据,即150*5,+1是将当天的开盘价引入作为一条特征数据
x = np.zeros((data.shape[0] - dayfeature, featurenum + 1))
y = np.zeros((data.shape[0] - dayfeature))  # 记录涨跌


for i in range(0,data.shape[0]-dayfeature):
    # 将当天的前150天的收盘价、最高价、最低价、开盘价、成交量放入x的1到featurenum列之中
    x[i, 0:featurenum] = np.array(data[i:i+dayfeature][[u'收盘价', u'最高价', u'最低价', u'开盘价', u'成交量']]).reshape((1, featurenum))
    # 将当天的开盘价也作为一个特征放入x的第fearturenum+1列中
    x[i, featurenum] = data.ix[i + dayfeature][u'开盘价']


for i in range(0, data.shape[0]-dayfeature):
    # 标记当天的涨跌情况
    if data.ix[i + dayfeature][u'收盘价'] >= data.ix[i + dayfeature][u'开盘价']:
        y[i] = 1
    else:
        y[i] = 0

# 调用svm函数,并设置kernel参数,默认是rbf,其它:'linear','poly','sigmoid'
clf = svm.SVC(kernel='rbf')
result = []
for i in range(5):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf.fit(x_train, y_train)
    result.append(np.mean(y_test == clf.predict(x_test)))  #五次交叉验证的结果存入result列表中

print('svm Classifier accuacy:')
print(result)
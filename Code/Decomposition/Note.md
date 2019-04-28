[TOC]



# fit_transform,fit,transform的区别

+ 参考https://blog.csdn.net/weixin_38278334/article/details/82971752
+ fit：其实不是一个train过程，而是一个适配的过程，简单来说，就是求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性
+ transform：在fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）
+ fit_transform：fit_transform是fit和transform的组合，既包括了训练又包含了转换
+ fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等



# fit，predict各是干什么的

+ 参考https://www.jianshu.com/p/731610dca805



sklearn中所有模型都有下列四个固定且常用的方法：

```python
# 拟合模型
model.fit(X_train, y_train)
# 模型预测
model.predict(X_test)
# 获得这个模型的参数
model.get_params()
# 为模型进行打分
model.score(data_X, data_y) # 回归问题：以R2参数为标准 分类问题：以准确率为标准
```



sklearn中的数据集也有一些常用用法：

```python
# 导入数据集
XXXdata = load_XXX()

# 数据集的feature
XXXdata.data

# 数据集的label
XXXdata.target
```







# numpy中随机数种子seed的含义

+ https://www.cnblogs.com/lutingting/p/5185408.html
+ 对于某个伪随机数发生器，只要seed不变产生的随机序列就是相同的



```python
In [20]: rdm = RandomState(100)

In [21]: print(rdm.uniform(0, 1, (2,2)))
[[0.54340494 0.27836939]
 [0.42451759 0.84477613]]

In [22]: rdm = RandomState(100)

In [23]: print(rdm.uniform(0, 1, (2,2)))
[[0.54340494 0.27836939]
 [0.42451759 0.84477613]]

In [24]: rdm = RandomState(100)

In [25]: print(rdm.uniform(0, 1, (2,2)))
[[0.54340494 0.27836939]
 [0.42451759 0.84477613]]
```

和下面：

```python
In [20]: rdm = RandomState(100)

In [21]: print(rdm.uniform(0, 1, (2,2)))
[[0.54340494 0.27836939]
 [0.42451759 0.84477613]]

In [22]: rdm = RandomState(100)

In [23]: print(rdm.uniform(0, 1, (2,2)))
[[0.54340494 0.27836939]
 [0.42451759 0.84477613]]

In [24]: rdm = RandomState(100)

In [25]: print(rdm.uniform(0, 1, (2,2)))
[[0.54340494 0.27836939]
 [0.42451759 0.84477613]]

```

是不一样的！！！注意区别





# 交叉验证

+ https://www.jianshu.com/p/dbc84ac47bc7
+ 参考：https://www.jianshu.com/p/00f5b4376c9a
  + 交叉验证返回的每次的得分score，和上面的model.score()函数应该是一样的，只不过因为分成了多组，会有多个score



# 导入train_test_split

+ https://www.jianshu.com/p/d746c9e10b2f



# classification_report

+ 会返回什么：https://blog.csdn.net/akadiao/article/details/78788864



# np.ravel()和np.flatten()

+ 参考：https://blog.csdn.net/hanshuobest/article/details/78882425
+ 两者都是将多维数组降为一维
+ 但是np.flatten()不会对原始矩阵有影响，但np.ravel()对原始矩阵会有影响



# train_test_split()函数解析用法

+ https://blog.51cto.com/12831900/2300061
+ https://www.cnblogs.com/bonelee/p/8036024.html
+ https://blog.csdn.net/kyriehe/article/details/77507473







# cross_validation包无法使用

+ https://blog.csdn.net/qq_37054356/article/details/83627669
+ 直接使用`from skleearn.model_selection import train_test_split`即可
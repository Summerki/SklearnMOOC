[TOC]

# Cluster之KMeans

>  关于sklearn中Cluster之KMeans的使用可以参考博客：https://blog.csdn.net/sinat_26917383/article/details/70240628



# 关于fit、predict、fit_predict三者的区别

+ fit：自己取的模型名称.fit(data)==>使用自己的模型训练数据
+ predict：自己的模型名称.predict(data)==>得到使用自己的模型预测data数据的标签，可用一个变量进行接收
+ fit_predict：相当于是fit+predict，省去了中间过程



# Cluster之DBSCAN算法

+ 为了解决KMeans只能处理球形的簇的局限性，于是就有了DBSCAN算法，可以聚类出各种环形、不规则图形
+ DBSCAN概念参考博客：[参考](https://blog.csdn.net/huacha__/article/details/81094891)
  + 由于我直接复制博客链接地址跳转不了所以使用上面超链接方法
+ 使用方法参考博客：https://blog.csdn.net/sinat_26917383/article/details/74932608





# np.array输出(2,)和(2,2)有什么区别？

+ https://zhidao.baidu.com/question/243724134053649164.html



# np.reshape((-1,1))是什么意思？

+ https://blog.csdn.net/wld914674505/article/details/80460042
+ 数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值



# pycharm关于matploylib作图弹出的问题

+ https://blog.csdn.net/u010472607/article/details/82290159
+ 
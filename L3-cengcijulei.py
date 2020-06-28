# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:27:25 2020

@author: baochao
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:08:49 2020

@author: baochao
"""


# 使用KMeans进行聚类
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

# 数据加载
data = pd.read_csv('car.csv', encoding='gbk')

train_x = data[["人均GDP","城镇人口比重","交通工具消费价格指数", "百户拥有汽车量"]]



# 规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
#pd.DataFrame(train_x).to_csv('temp.csv', index=False)
#print(train_x)



from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
model = AgglomerativeClustering(linkage='ward', n_clusters=3)
y = model.fit_predict(train_x)
print(y)
linkage_matrix = ward(train_x)
dendrogram(linkage_matrix)
plt.show()



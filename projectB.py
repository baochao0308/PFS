# -*- coding: utf-8 -*-
#表格只留下客户ID和产品名称，然后对表格数据按照客户ID进行升序排列
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_apriori import apriori
import time
start = time.time()

dataset = pd.read_csv('./订单表.csv',encoding='gbk') 

# 将数据存放到transactions中
transactions = []
a = dataset.values[0,0]
temp = []
for i in range(0, dataset.shape[0]):
    
    if str(dataset.values[i,0]) == str(a):
        temp.append(str(dataset.values[i, 1]))
    else:
        a = dataset.values[i,0]
        temp = set(temp)
        transactions.append(temp)
        temp = [dataset.values[i, 1]]



print(transactions)
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.01,  min_confidence=0.5)
print("频繁项集：", itemsets)
print("关联规则：", rules)
end = time.time()
print("用时：", end-start)
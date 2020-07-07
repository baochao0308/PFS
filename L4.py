import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_apriori import apriori


dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header = None) 

print(dataset.shape)


transactions = []
for i in range(0, dataset.shape[0]):
    temp = []
    for j in range(0, 20):
        if str(dataset.values[i, j]) != 'nan':
           temp.append(str(dataset.values[i, j]))
    transactions.append(temp)

itemsets, rules = apriori(transactions, min_support=0.04,  min_confidence=0.2)
print("频繁项集：", itemsets)
print("关联规则：", rules)

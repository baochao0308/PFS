# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:49:36 2020

@author: baochao
"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
data = {'Chinese': [66, 95, 93, 90,80], 'Math': [30, 98, 96, 77, 90], 'English': [65, 85, 92, 88, 90]}
df1 = DataFrame(data)
df2 = DataFrame(data, index=['张飞', '关羽', '刘备', '典韦', '许褚'], columns=['Chinese', 'Math', 'English'])
df2.rename(columns={'Chinese': '语文', 'Math': '数学','English': '英语'}, inplace = True)
'''
print(df2.min())
print(df2.max())
print(df2.mean())
print(df2.var())
print(df2.std())
'''
print(df2.agg(['max','min','mean','var','std']))

df2['总计'] = df2.apply(lambda x: x.sum(), axis=1)
df2=df2.sort_values('总计',ascending=False)
print(df2)








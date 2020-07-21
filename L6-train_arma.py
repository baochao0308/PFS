# -*- coding: utf-8 -*-
# 沪市指数走势预测，使用时间序列ARMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
import warnings
from itertools import product
from datetime import datetime, timedelta
import calendar

warnings.filterwarnings('ignore')
# 数据加载
df = pd.read_csv('./train.csv')
df = df[['Datetime', 'Count']]
#df = df.resample('D').sum()
#print(df.head())


# 将时间作为df的索引
df.Datetime = pd.to_datetime(df.Datetime)

df.index = df.Datetime
# 数据探索
print(df.head())
# 按照月，季度，年来统计

df_month = df.resample('M').sum()
df_Q = df.resample('Q-DEC').sum()
df_year = df.resample('A-DEC').sum()
#print(df_month)
#print(df_Q)
#print(df_year)

# 按照天，月，季度，年来显示沪市指数的走势
fig = plt.figure(figsize=[15, 7])
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.suptitle('沪市指数', fontsize=20)
plt.subplot(221)
plt.plot(df.Count, '-', label='按天')
plt.legend()
plt.subplot(222)
plt.plot(df_month.Count, '-', label='按月')
plt.legend()
plt.subplot(223)
plt.plot(df_Q.Count, '-', label='按季度')
plt.legend()
plt.subplot(224)
plt.plot(df_year.Count, '-', label='按年')
plt.legend()
plt.show()

# 设置参数范围
ps = range(0, 2)
qs = range(0, 2)
parameters = product(ps, qs)
parameters_list = list(parameters)
# 寻找最优ARMA模型参数，即best_aic最小
results = []
best_aic = float("inf") # 正无穷
for param in parameters_list:
    try:
        model = ARMA(df.Count,order=(param[0], param[1])).fit()
    except ValueError:
        print('参数错误:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# 输出最优模型
print('最优模型: ', best_model.summary())

# 设置future_month，需要预测的时间date_list
df_month2 = df_month[['Count']]
future_month = 7
last_month = pd.to_datetime(df_month2.index[len(df_month2)-1])
#print(last_month)
date_list = []
for i in range(future_month):
    # 计算下个月有多少天
    year = last_month.year
    month = last_month.month
    if month == 12:
        month = 1
        year = year+1
    else:
        month = month + 1
    next_month_days = calendar.monthrange(year, month)[1]
    #print(next_month_days)
    last_month = last_month + timedelta(days=next_month_days)
    date_list.append(last_month)
print('date_list=', date_list)

# 添加未来要预测的7个月
future = pd.DataFrame(index=date_list, columns= df_month.columns)
df_month2 = pd.concat([df_month2, future])
df_month2['forecast'] = best_model.predict(start=0, end=len(df_month2))
# 第一个元素不正确，设置为NaN
df_month2['forecast'][0] = np.NaN
print(df_month2)

# 预测结果显示
plt.figure(figsize=(30,7))
df_month2.Count.plot(label='实际客流量')
df_month2.forecast.plot(color='r', ls='--', label='预测客流量')
plt.legend()
plt.title('客流量（月）')
plt.xlabel('时间')
plt.ylabel('客流量')
plt.show()

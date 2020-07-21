#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
train = pd.read_csv('train.csv')
print(train.head())


# In[5]:


train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index = train.Datetime
print(train.head())


# In[6]:


train.drop(['ID','Datetime'],axis=1, inplace=True)
print(train.head())


# In[8]:


daily_train = train.resample('D').sum()
print(daily_train.head())
daily_train['ds'] = daily_train.index
daily_train['y'] = daily_train.Count
daily_train.drop(['Count'],axis=1, inplace=True)
print(daily_train.head())


# In[10]:

from fbprophet import Prophet
m = Prophet(yearly_seasonality=Ture,seasonality_prior_scale=0.1)
m.fit(daily_train)
future = m.make_future_dataframe(periods=213)
forecast = m.predict(future)
print(forecast)


# In[1]:


m.plot(forecast)


# In[ ]:


m.plot_components(forecast)


# 对汽车投诉信息进行分析
import pandas as pd

result = pd.read_csv('car_complain.csv')
#print(result)
# 将genres进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
result = result.drop('problem', 1).join(result.problem.str.get_dummies(','))
#print(result.columns)
tags = result.columns[7:]
#print(tags)

df= result.groupby(['brand'])['id'].agg(['count']).sort_values('count',ascending=False)
print(df)
df3= result.groupby(['car_model'])['id'].agg(['count']).sort_values('count',ascending=False)
print(df3)

df4= result.groupby(['brand','car_model'])['id'].agg(['count']).sort_values('count',ascending=False).groupby(['brand']).mean().sort_values('count',ascending=False)
print(df4)




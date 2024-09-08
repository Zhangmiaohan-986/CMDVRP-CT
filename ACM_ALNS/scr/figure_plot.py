import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '/Users/zhangmiaohan/PycharmProjects/gurobi_test/tits_CMDVRP-CT/map/R201.txt'
with open(file_path, 'r') as file:
    data0 = file.read()

# 将数据转换为DataFrame
data = []
for line in data0.strip().split('\n'):
    data.append(line.split())

columns = ["CUST", "XCOORD.", 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
df = pd.DataFrame(data[1:26], columns=columns)

# 将字符型列转换为数字
numeric_cols = ['CUST', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df['sum'] = df['DUE'] - df['READY']
print(np.mean(df['sum']))

# """画时间窗分布图"""
# plt.barh(df.index, df['DUE'] - df['READY'], left=df['READY'], color='blue', alpha=0.7)
# # 显示图形
# plt.show()

"""画散点图：运行时去掉注释"""
# 绘制散点图
plt.scatter(df.loc[1:, 'XCOORD.'], df.loc[1:, 'YCOORD.'], label='Customers', marker='o', color='blue')
plt.scatter(df.loc[0, 'XCOORD.'], df.loc[0, 'YCOORD.'], label='Depot', marker='x', color='red')
# 添加标签
for i, row in df.iterrows():
    plt.annotate(row['CUST'], (row['XCOORD.'], row['YCOORD.']), textcoords="offset points", xytext=(0, 5), ha='center')
# 添加标题和轴标签
plt.title(f'Customer and Depot Locations of {file_path}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
# 显示图例
plt.legend()
# 显示图形
plt.show()
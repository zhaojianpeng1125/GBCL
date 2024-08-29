import pandas as pd

# 读取没有列名的CSV文件
df = pd.read_csv('Dataset/IMDB/train_vec.csv', header=None)

# 检查数据框的前几行以确认读取成功
print("Before moving the first column to the last position:")
print(df.head())

# 获取第一列的数据
first_column = df.iloc[:, 0]

# 移除第一列并将其添加到最后一列
df = df.drop(columns=[0])
df[len(df.columns)] = first_column

# 保存修改后的CSV文件
df.to_csv('imdb/train_vec.csv', index=False, header=False)

# 检查数据框的前几行以确认移动成功
print("After moving the first column to the last position:")
print(df.head())

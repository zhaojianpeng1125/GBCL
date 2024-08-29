import pandas as pd

# 读取新的 CSV 文件
df = pd.read_csv('train_processed.csv')

# 计算行数
row_count = len(df)

print(f'The file contains {row_count} rows.')

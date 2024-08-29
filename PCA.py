import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# 读取CSV文件
data = pd.read_csv('train_vec.csv')

# 将最后一列作为标签
labels = data.iloc[:, -1]
features = data.iloc[:, :-1]

# 使用PCA将特征降维到40维
pca = PCA(n_components=40)
reduced_features = pca.fit_transform(features)

# 将降维后的特征和标签合并
reduced_data = np.hstack((reduced_features, labels.values.reshape(-1, 1)))

# 转换为DataFrame
reduced_df = pd.DataFrame(reduced_data)

# 保存降维后的数据到新的CSV文件
reduced_df.to_csv('reduced_noise_10%_train_features.csv', index=False, header=True)

print("PCA降维完成，结果已保存到'reduced_noise_10%_train_features.csv'")

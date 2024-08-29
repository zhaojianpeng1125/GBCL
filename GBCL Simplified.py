import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import time  # 引入 time 模块
# 忽略所有警告
warnings.filterwarnings("ignore")

class GrainCluster:
    def __init__(self, features, labels, chunk_start, purity_threshold=0.9):
        self.features = features
        self.labels = labels
        self.purity_threshold = purity_threshold
        self.chunk_start = chunk_start
        self.updated_labels = np.copy(labels)

    def calculate_purity(self, labels):
        if len(labels) == 0:
            return 1
        most_common_count = Counter(labels).most_common(1)[0][1]
        return most_common_count / len(labels)

    def process_initial_cluster(self):
        initial_purity = self.calculate_purity(self.labels)
        if initial_purity < self.purity_threshold:
            self.split_clusters(self.features, self.labels, np.arange(len(self.labels)))

    def split_clusters(self, features, labels, indices):
        splitting_queue = [(features, labels, indices)]
        updated_labels = np.copy(labels)
        processed_count = 0

        while splitting_queue:
            if processed_count % 20 == 0:
                print(f"待分裂队列中粒球的个数: {len(splitting_queue)}")

            current_features, current_labels, indices = splitting_queue.pop(0)
            processed_count += 1
            unique_labels = np.unique(current_labels)
            k = len(unique_labels)

            if k == 0:
                print("警告：当前分裂的簇没有任何标签！")
                continue

            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
            clusters = kmeans.fit_predict(current_features)

            for i in range(k):
                cluster_indices = np.where(clusters == i)[0]
                cluster_features = current_features[cluster_indices]
                cluster_labels = current_labels[cluster_indices]
                cluster_purity = self.calculate_purity(cluster_labels)

                if cluster_purity >= self.purity_threshold:
                    most_common = Counter(cluster_labels).most_common(1)
                    if most_common:  # 确保most_common不为空
                        most_common_label = most_common[0][0]
                        updated_labels[indices[cluster_indices]] = most_common_label
                else:
                    splitting_queue.append((cluster_features, cluster_labels, indices[cluster_indices]))

        self.updated_labels = updated_labels

    def get_updated_labels(self):
        return self.updated_labels

start_time = time.time()

# 外部循环处理数据块
data_path = 'new_dataset/20%_asymmetric_noisy_490K_train_vec.csv'
original_data_path = 'new_dataset/20%_asymmetric_noisy_490K_train.csv'
output_path = 'new_dataset/20%_asymmetric_noisy_corrected_490K_train.csv'
chunk_size = 2000

updated_labels = []

# 读取原始数据标签
original_data = pd.read_csv(original_data_path)
original_labels = original_data.iloc[:, -1].values

chunk_iter = pd.read_csv(data_path, chunksize=chunk_size)
chunk_start = 0

for chunk_number, chunk in enumerate(chunk_iter, start=1):
    features = chunk.iloc[:, :-1].values
    labels = chunk.iloc[:, -1].values

    grain_cluster = GrainCluster(features, labels, chunk_start, purity_threshold=0.91)
    grain_cluster.process_initial_cluster()
    updated_chunk_labels = grain_cluster.get_updated_labels()

    updated_labels.extend([(chunk_start + idx, label) for idx, label in enumerate(updated_chunk_labels)])
    chunk_start += len(chunk)

    # 打印提示信息
    print(f"已处理批次 {chunk_number}, 处理行数 {chunk_start}")

# 更新原始数据标签并保存
for index, label in updated_labels:
    original_labels[index] = label

original_data.iloc[:, -1] = original_labels
original_data.to_csv(output_path, index=False)
print(f"修正后的标签已保存到 {output_path}")

# 记录结束时间
end_time = time.time()

# 打印运行时间
print(f"代码运行时间: {end_time - start_time:.2f}秒")

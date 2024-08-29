import numpy as np
import pandas as pd
import random
import time  # 引入 time 模块

def calculate_distances(data, p):
    """
    计算加权欧式距离
    """
    dis = (data - p)**2
    dis_top10 = np.sort(dis, axis=0)[-10:]
    return 0.6 * (dis**0.5).sum() + 0.4 * (dis_top10**0.5).sum()

class GrainCluster:
    def __init__(self, purity_threshold=0.9):
        self.purity_threshold = purity_threshold  # 设定纯度阈值
        self.to_process_queue = []  # 初始化待处理的粒球队列
        self.processed_grains = []  # 初始化已处理的粒球列表

    def load_data(self, filepath):
        data = pd.read_csv(filepath)  # 读取CSV文件
        self.features = data.iloc[:, :-1].values  # 提取特征（去掉最后一列）
        self.labels = data.iloc[:, -1].values  # 提取标签（最后一列）
        initial_grain = (self.features, self.labels)  # 创建初始粒球
        self.to_process_queue.append(initial_grain)  # 将初始粒球加入待处理队列

    def calculate_purity(self, labels):
        labels = np.asarray(labels).flatten()  # 确保labels为一维数组
        label_counts = np.bincount(labels)  # 统计每个标签的数量
        max_label_count = np.max(label_counts)  # 找到数量最多的标签数
        purity = max_label_count / len(labels)  # 计算纯度
        return purity, np.argmax(label_counts)  # 返回纯度和占比最大的标签

    def split_grain(self, features, labels):
        unique_labels = np.unique(labels)  # 找到所有唯一的标签
        k = random.randint(1, len(unique_labels))
        centers = features[np.random.choice(range(len(features)), k, replace=False)]  # 随机选择k个样本作为中心点
        distances = np.array([[calculate_distances(feature, center) for feature in features] for center in centers])  # 计算所有样本到中心点的加权欧式距离
        new_grains = []  # 初始化新粒球列表
        for i in range(k):
            grain_indices = np.argmin(distances, axis=0) == i  # 找到距离第i个中心点最近的样本
            new_features = features[grain_indices]  # 提取这些样本的特征
            new_labels = labels[grain_indices]  # 提取这些样本的标签
            new_grains.append((new_features, new_labels))  # 创建新的粒球并加入列表
        return new_grains  # 返回新的粒球列表

    def process_grains(self):
        while self.to_process_queue:  # 当待处理队列不为空时
            print(f"待处理的粒球个数: {len(self.to_process_queue)}")  # 打印待处理的粒球个数
            features, labels = self.to_process_queue.pop(0)  # 取出队列中的第一个粒球
            purity, dominant_label = self.calculate_purity(labels)  # 计算粒球的纯度和占比最大的标签
            if purity >= self.purity_threshold:  # 如果纯度满足要求
                self.processed_grains.append((features, np.full(len(labels), dominant_label)))  # 将粒球标记为已处理，并将所有样本的标签设为占比最大的标签
            else:
                new_grains = self.split_grain(features, labels)  # 否则分裂粒球
                self.to_process_queue.extend(new_grains)  # 将分裂出的新粒球加入待处理队列

    def correct_labels(self):
        corrected_labels = np.zeros_like(self.labels)  # 初始化修正后的标签数组
        for features, labels in self.processed_grains:  # 遍历所有已处理的粒球
            for i in range(len(features)):  # 遍历每个粒球中的样本
                index = np.where((self.features == features[i]).all(axis=1))[0][0]  # 找到样本在原始数据中的索引
                corrected_labels[index] = labels[i]  # 修正标签
        return corrected_labels  # 返回修正后的标签

    def save_corrected_data(self, original_filepath, corrected_labels, output_filepath):
        original_data = pd.read_csv(original_filepath)  # 读取原始数据
        original_data.iloc[:, -1] = corrected_labels  # 替换原始标签为修正后的标签
        original_data.to_csv(output_filepath, index=False)  # 保存为新的CSV文件

# 记录开始时间
start_time = time.time()

# 使用示例
gc = GrainCluster(purity_threshold=0.95)  # 创建GrainCluster实例，设置纯度阈值为0.9
gc.load_data('split_dataset/30%_noisy_train_vector.csv')  # 加载数据
gc.process_grains()  # 处理粒球
corrected_labels = gc.correct_labels()  # 修正标签
gc.save_corrected_data('split_dataset/train1.csv', corrected_labels, 'split_dataset/30%_noise_corrected_train.csv')  # 保存修正后的数据

# 记录结束时间
end_time = time.time()

# 打印运行时间
print(f"代码运行时间: {end_time - start_time:.2f}秒")

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# 加载数据
df = pd.read_csv('../GBCL_Amazon-5/Dataset/Amazon-5/Original dataset/train.csv')
df['label'] = df['label'].astype(int)

def generate_symmetric_noise(df, noise_rate):
    labels = df['label'].unique()
    num_labels = len(labels)

    noisy_df = df.copy()

    num_noisy_samples = int(noise_rate * len(df))

    for i in range(num_noisy_samples):
        original_label = noisy_df.loc[i, 'label']
        noisy_label = original_label
        while noisy_label == original_label:
            noisy_label = np.random.choice(labels)
        noisy_df.loc[i, 'label'] = noisy_label

    return shuffle(noisy_df)


def generate_asymmetric_noise(df, noise_rate):
    labels = df['label'].unique()

    noisy_df = df.copy()

    num_noisy_samples = int(noise_rate * len(df))

    for i in range(num_noisy_samples):
        original_label = noisy_df.loc[i, 'label']
        # 假设标签是0,1,...,N-1, 非对称噪声标签可以是 original_label + 1 或者原标签 + 1 mod N
        noisy_label = (original_label + 1) % len(labels)
        noisy_df.loc[i, 'label'] = noisy_label

    return shuffle(noisy_df)


# 生成对称噪声数据集
symmetric_noise_0_1 = generate_symmetric_noise(df, 0.2)
symmetric_noise_0_2 = generate_symmetric_noise(df, 0.4)
symmetric_noise_0_3 = generate_symmetric_noise(df, 0.6)
symmetric_noise_0_4 = generate_symmetric_noise(df, 0.8)

# 生成非对称噪声数据集
asymmetric_noise_0_2 = generate_asymmetric_noise(df, 0.2)
asymmetric_noise_0_4 = generate_asymmetric_noise(df, 0.4)

# 保存结果
symmetric_noise_0_1.to_csv('Dataset/Amazon-5/Noisy dataset/symmetric_noise_0_2.csv', index=False)
symmetric_noise_0_2.to_csv('Dataset/Amazon-5/Noisy dataset/symmetric_noise_0_4.csv', index=False)
symmetric_noise_0_3.to_csv('Dataset/Amazon-5/Noisy dataset/symmetric_noise_0_6.csv', index=False)
symmetric_noise_0_4.to_csv('Dataset/Amazon-5/Noisy dataset/symmetric_noise_0_8.csv', index=False)

asymmetric_noise_0_2.to_csv('Dataset/Amazon-5/Noisy dataset/asymmetric_noise_0_2.csv', index=False)
asymmetric_noise_0_4.to_csv('Dataset/Amazon-5/Noisy dataset/asymmetric_noise_0_4.csv', index=False)

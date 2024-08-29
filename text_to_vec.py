import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 加载训练好的BERT模型
model_path = "../../model_save/Yelp-5_best_model.bin"
model = SentimentClassifier(n_classes=5)
model = nn.DataParallel(model)  # 这一步是必要的，因为您训练时使用了DataParallel
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.module  # 这一步是必要的，因为我们需要从DataParallel中提取原始模型
model.to(device)
model.eval()

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载数据集
data_path = "../../Dataset/YELP-5/Original dataset/train.csv"
data = pd.read_csv(data_path, encoding='utf-8')

# 将文本转换为BERT特征向量
def text_to_features(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.bert(**inputs)  # 使用基础的BERT模型部分
    # 获取 [CLS] token 的特征向量
    cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_features

batch_size = 1000  # 每处理1000条数据打印一次提醒
features_list = []
for i, text in enumerate(data['text']):
    features = text_to_features(text, tokenizer, model, device)
    features_list.append(features.squeeze())  # 去掉多余的维度
    if (i + 1) % batch_size == 0:
        print(f'已处理 {i + 1} 条数据')

features_df = pd.DataFrame(features_list)
features_df['label'] = data['label']

# 保存新数据集
features_path = "../Dataset/YELP-5/train_vec.csv"
features_df.to_csv(features_path, index=False, header=True)

print(f"特征向量已保存到 {features_path}")

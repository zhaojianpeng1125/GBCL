import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import numpy as np
import re
import random
import os
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

def load_and_preprocess_data(train_path, val_path, test_path):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        return text

    train_data['cleaned_text'] = train_data['text'].apply(clean_text)
    val_data['cleaned_text'] = val_data['text'].apply(clean_text)
    test_data['cleaned_text'] = test_data['text'].apply(clean_text)

    label_encoder = LabelEncoder()
    train_data['label'] = label_encoder.fit_transform(train_data['label'])
    val_data['label'] = label_encoder.transform(val_data['label'])
    test_data['label'] = label_encoder.transform(test_data['label'])

    return train_data, val_data, test_data

def create_datasets_and_loaders(train_data, val_data, test_data, tokenizer, max_len, batch_size):
    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }

    train_dataset = SentimentDataset(train_data['cleaned_text'].tolist(), train_data['label'].tolist(), tokenizer, max_len)
    val_dataset = SentimentDataset(val_data['cleaned_text'].tolist(), val_data['label'].tolist(), tokenizer, max_len)
    test_dataset = SentimentDataset(test_data['cleaned_text'].tolist(), test_data['label'].tolist(), tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

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

def train_and_evaluate(train_loader, val_loader, test_loader, model_save_path, history_save_path, epochs=10, lr=2e-5):
    model = SentimentClassifier(n_classes=5)
    model = nn.DataParallel(model)  # 数据并行
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()  # 混合精度训练

    def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
        model = model.train()
        losses = []
        correct_predictions = 0

        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['label'].to(device)

            optimizer.zero_grad()

            with autocast():  # 自动混合精度
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    def eval_model(model, data_loader, loss_fn, device, n_examples):
        model = model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                labels = d['label'].to(device)

                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, preds = torch.max(outputs, dim=1)
                    loss = loss_fn(outputs, labels)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    best_accuracy = 0
    history = {'train_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, len(train_loader.dataset))
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device, len(val_loader.dataset))
        print(f'Val loss {val_loss} accuracy {val_acc}')

        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_acc.item())

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_save_path, index=False)

    model.load_state_dict(torch.load(model_save_path))

    test_acc, test_loss = eval_model(model, test_loader, loss_fn, device, len(test_loader.dataset))
    print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# symmetric_noise_0_2
train_data, val_data, test_data = load_and_preprocess_data(
    '../../purity_test/Dataset/SST-5/Corrected dataset/GBCL_Simplified_Corrected/symmetric_noise_0_2.csv',
    '../../purity_test/Dataset/SST-5/Original dataset/val.csv',
    '../../purity_test/Dataset/SST-5/Original dataset/test.csv')
train_loader, val_loader, test_loader = create_datasets_and_loaders(train_data, val_data, test_data, tokenizer, max_len=100, batch_size=256)
train_and_evaluate(train_loader, val_loader, test_loader, '../../model_save/test.bin',
                   '../../purity_test/Dataset/SST-5/res/Simplified/symmetric_noise_0_1.csv')

# symmetric_noise_0_2
train_data, val_data, test_data = load_and_preprocess_data(
    '../../purity_test/Dataset/SST-5/Corrected dataset/GBCL_Prime_Corrected/symmetric_noise_0_2.csv',
    '../../purity_test/Dataset/SST-5/Original dataset/val.csv',
    '../../purity_test/Dataset/SST-5/Original dataset/test.csv')
train_loader, val_loader, test_loader = create_datasets_and_loaders(train_data, val_data, test_data, tokenizer, max_len=100, batch_size=256)
train_and_evaluate(train_loader, val_loader, test_loader, '../../model_save/test.bin',
                   '../../purity_test/Dataset/SST-5/res/Prime/symmetric_noise_0_2.csv')


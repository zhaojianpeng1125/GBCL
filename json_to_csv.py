import json
import csv

filein = 'user_dedup.json'
fileout = 'train1.csv'
batch_size = 2000  # 每次处理的行数
max_lines = 3000000  # 只转换前max_lines行

def process_batch(batch, writer):
    for data in batch:
        writer.writerow(data)

headers_written = False
line_count = 0

with open(filein, encoding='utf-8') as jsonf, open(fileout, 'w', newline='', encoding='utf-8') as csvf:
    writer = None
    batch = []
    for line in jsonf:
        line = line.strip()  # 移除首尾的空白字符
        if not line:
            continue  # 跳过空行
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e} - Line: {line}")
            continue  # 跳过解析错误的行
        if not headers_written:
            # Initialize writer with fieldnames from the first line's keys
            writer = csv.DictWriter(csvf, fieldnames=data.keys())
            writer.writeheader()
            headers_written = True
        batch.append(data)
        line_count += 1
        if line_count >= max_lines:
            break
        if len(batch) == batch_size:
            process_batch(batch, writer)
            batch = []  # 清空批次
    if batch:
        process_batch(batch, writer)  # 处理最后一批

print(f'done! Processed {line_count} lines.')

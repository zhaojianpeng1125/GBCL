import pandas as pd

def remove_non_utf8_rows(file_path):
    valid_rows = []
    total_rows = 0

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            total_rows += 1
            if '\ufffd' not in line:  # '\ufffd' is the replacement character used for encoding errors
                valid_rows.append(line)

    # Write valid rows back to a new CSV file
    with open('noise_10%_train.csv', 'w', encoding='utf-8') as file:
        file.writelines(valid_rows)

    print(f"Total valid rows: {len(valid_rows)}")
    print(f"Total rows processed: {total_rows}")

if __name__ == "__main__":
    remove_non_utf8_rows('test1.csv')

import pandas as pd

def load_and_check_data(filepath):
    """
    加载CSV文件并检查是否有缺失值
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"文件 {filepath} 未找到，请检查文件路径。")
        return
    except pd.errors.EmptyDataError:
        print(f"文件 {filepath} 为空，请检查文件内容。")
        return
    except pd.errors.ParserError:
        print(f"文件 {filepath} 解析错误，请检查文件格式。")
        return

    print(f"数据文件 {filepath} 加载成功！")

    # 检查每列的缺失值情况
    missing_values = data.isnull().sum()
    total_missing = missing_values.sum()

    if total_missing == 0:
        print("数据文件中没有缺失值。")
    else:
        print("数据文件中存在缺失值。")
        print("每列的缺失值情况如下：")
        print(missing_values)

    return data

if __name__ == "__main__":
    filepath = '/GBCL/Dataset/SST-5\Original Dataset\SST5.csv'  # 将此处替换为你的数据文件路径
    load_and_check_data(filepath)

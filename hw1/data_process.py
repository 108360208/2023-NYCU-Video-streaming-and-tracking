import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始CSV文件
data = pd.read_csv("./dataset/train.csv")  # 假设数据以制表符分隔

# 确定验证集占总数据的比例
valid_ratio = 0.2

# 按照类别对数据进行分组
grouped = data.groupby("label")

# 初始化用于存储新训练集和验证集的列表
new_train_data = []
new_valid_data = []

# 对每个类别进行拆分
for label, group in grouped:
    # 将当前类别的数据分为新的训练集和验证集
    train_group, valid_group = train_test_split(group, test_size=valid_ratio, random_state=42)

    # 将拆分后的数据添加到新的训练集和验证集列表中
    new_train_data.append(train_group)
    new_valid_data.append(valid_group)

# 合并所有类别的数据以创建最终的新训练集和验证集
new_train_data = pd.concat(new_train_data)
new_valid_data = pd.concat(new_valid_data)

# 保存新的训练集和验证集为CSV文件
new_train_data.to_csv("new_train.csv", index=False)
new_valid_data.to_csv("new_valid.csv", index=False)

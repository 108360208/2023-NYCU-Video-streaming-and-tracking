import pandas as pd

# 读取两个 CSV 文件
df1 = pd.read_csv(r"C:\Users\Steven\Downloads\pred_312551093.csv")
df2 = pd.read_csv('pred_312581006.csv')
# 比较两个数据框的'label'列
different_rows = df1[df1['label'] != df2['label']]

# 显示不同的行
print("不同的行：")
print(len(different_rows))
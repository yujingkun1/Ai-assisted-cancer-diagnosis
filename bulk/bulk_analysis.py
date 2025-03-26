import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
df_target = pd.read_excel('target_bulk.xlsx')
df_target.columns = ['gene_id'] + list(df_target.columns[1:])
df_target.set_index('gene_id', inplace=True)
samples = df_target.columns[0]



'''气泡图'''

# 从大到小排序
values = df_target[samples].sort_values(ascending=False)[:100]
# 取log10
value = np.log10(values + 1)
gene_ids = value.index

# 创建气泡图
plt.figure(figsize=(6, 20))

# 选择不同的颜色，并反转颜色映射，使用 viridis_r 颜色映射
colors = plt.cm.viridis_r(np.linspace(0, 1, len(value)))  # viridis_r 颜色映射，颜色顺序反转

# 绘制气泡图，调整气泡大小和颜色
plt.scatter(value, gene_ids, s=value * 10, alpha=0.8, c=colors)  # 气泡大小调整为原来的 100 倍
plt.colorbar(label='log10(TPM)')  # 添加颜色条

# 将y轴刻度逆序排列
plt.gca().invert_yaxis()

plt.title('Top 100 genes in sample')
plt.xlabel('log10(TPM)')
plt.ylabel('Gene ID')
plt.savefig(r'output\top_100_genes.png', dpi=300)



'''直方图'''
# 取出每个行索引的前15个字符
df_target.index = df_target.index.str[:15]
# 直方图
plt.figure(figsize=(8, 6))
# 选取非零值
df_target_nonzero = df_target[df_target > 0]
plt.hist(df_target_nonzero, bins=100, log=True)
plt.title('Histogram of gene expression values')
plt.xlabel('Expression value')
plt.ylabel('Frequency')
plt.savefig(r'output\gene_expression_histogram（去除所有0项）.png', dpi=300)



'''基础的统计学分析'''
specific_sample_data = df_target
mean_expression = specific_sample_data.mean()
median_expression = specific_sample_data.median()
std_expression = specific_sample_data.std()
min_expression = specific_sample_data.min()
max_expression = specific_sample_data.max()
print(f"均值: {mean_expression}")
print(f"中位数: {median_expression}")
print(f"标准差: {std_expression}")
print(f"最小值: {min_expression}")
print(f"最大值: {max_expression}")



'''导入数据'''
df = pd.read_csv('COAD_expression_tpm.csv')
# 设置第一列的列名
df.columns = ['gene_id'] + list(df.columns[1:])
# 设置第一列为索引
df.set_index('gene_id', inplace=True)
# 选取交集的基因
df = df[df.median(axis=1) > 1]
df_target = df_target.loc[df.index.intersection(df_target.index)]
df = df.loc[df.index.intersection(df_target.index)]
# 去重
df = df.loc[~df.index.duplicated(keep='first')]
df = df.loc[:, df.columns.str.contains('-11A-')]
df_target = df_target.loc[~df_target.index.duplicated(keep='first')]


'''数据处理'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# 假设 df_target 和 df 已经加载
# df_target 是特定病人的基因向量
# df 是多个病人的基因矩阵

# 确保基因 ID 对齐
common_genes = df.index.intersection(df_target.index)
df = df.loc[common_genes]
df_target = df_target.loc[common_genes]

# 计算每个基因的均值和标准差
gene_mean = df.mean(axis=1)
gene_std = df.std(axis=1)

import pandas as pd
import numpy as np

# 假设 df 和 df_target 已经加载
# df 是多个病人的基因矩阵，df_target 是特定病人的基因向量
# common_genes 是两个数据集中都有的基因列表

# 计算每个基因的均值和标准差
gene_mean = df.mean(axis=1)
gene_std = df.std(axis=1)

# 创建一个空的 DataFrame 来存储结果
result = pd.DataFrame(columns=['Gene', 'Normal Range', 'Target Value', 'Abnormality'])

# 遍历每个基因
import pandas as pd

# 假设 gene_mean 和 gene_std 是 Series，df_target 是 DataFrame，且它们的索引是基因名
# common_genes 是一个包含基因名的列表

# 创建一个空的 DataFrame 用于存储结果
result = pd.DataFrame(columns=['Gene', 'Normal Range', 'Target Value', 'Abnormality'])

# 计算正常范围
normal_ranges = pd.DataFrame({
    'Lower': gene_mean - 3 * gene_std,
    'Upper': gene_mean + 3 * gene_std
})

# 获取特定病人的基因值
target_values = df_target.loc[common_genes]

# 合并正常范围和目标值
merged = pd.concat([normal_ranges, target_values], axis=1)
merged.columns = ['Lower', 'Upper', 'Target Value']

# 判断异常情况
merged['Abnormality'] = 'Normal'
merged.loc[merged['Target Value'] < merged['Lower'], 'Abnormality'] = 'Significantly Less'
merged.loc[merged['Target Value'] > merged['Upper'], 'Abnormality'] = 'Significantly More'

# 筛选出异常的基因
abnormal_genes = merged[merged['Abnormality'] != 'Normal']

# 将结果添加到 DataFrame 中
result = abnormal_genes.reset_index()
result.columns = ['Gene', 'Lower', 'Upper', 'Target Value', 'Abnormality']
result['Normal Range'] = list(zip(result['Lower'], result['Upper']))
result = result[['Gene', 'Normal Range', 'Target Value', 'Abnormality']]

# 输出结果为 CSV 文件
result.to_csv(r'output\abnormal_genes.csv', index=False)
# 将结果绘制为箱线图，每种基因一个箱子，绘制正常的区间，并在图中标记异常值
import matplotlib.pyplot as plt
import seaborn as sns
# 绘制箱线图
# 仅仅绘制前100个基因
print(result)
result = result.head(50)
# 绘制箱线图
import matplotlib.pyplot as plt

# 数据
genes = result['Gene'].tolist()
normal_ranges = result['Normal Range'].tolist()
target_values = result['Target Value'].tolist()
# 创建箱线图
fig, ax = plt.subplots(figsize=(20, 12))

# 绘制箱线图
for i, gene in enumerate(genes):
    # 绘制 Normal Range
    ax.broken_barh([(normal_ranges[i][0], normal_ranges[i][1] - normal_ranges[i][0])], (i - 0.4, 0.8), facecolors='lightblue')
    # 特殊标识 Target Value
    ax.plot(target_values[i], i, 'ro', markersize=8, label='Target Value' if i == 0 else "")

# 设置坐标轴标签
ax.set_yticks(range(len(genes)))
ax.set_yticklabels(genes)
ax.set_xlabel('Value')
ax.set_title('Box Plot with Normal Range and Target Value')

# 添加图例
ax.legend()

# 显示图表
plt.tight_layout()
plt.savefig(r'output\abnormal_genes_boxplot.png', dpi=300)
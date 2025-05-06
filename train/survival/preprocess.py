import pandas as pd
import os
import numpy as np

# 指定 clinical_data 文件夹路径
clinical_dir = 'clinical_data/'  # 替换为你的实际路径，例如 '/home/yujk/survival_analysis/clinical_data/'

# 检查目录是否存在
if not os.path.exists(clinical_dir):
    print(f"错误：目录 {clinical_dir} 不存在！请检查路径。")
    exit()

# 收集所有 clinical_follow_up.txt 文件
follow_up_files = []
for root, dirs, files in os.walk(clinical_dir):
    for file in files:
        if 'clinical_follow_up' in file.lower() and file.endswith('.txt'):
            full_path = os.path.join(root, file)
            follow_up_files.append(full_path)
            print(f"找到文件: {full_path}")

# 检查是否找到文件
if not follow_up_files:
    print(f"错误：在 {clinical_dir} 中未找到任何 clinical_follow_up.txt 文件！")
    print("当前目录内容：")
    for root, dirs, files in os.walk(clinical_dir):
        print(f"子目录: {root}, 文件: {files}")
    exit()

# 合并数据
clinical_data = []
for file in follow_up_files:
    try:
        df = pd.read_csv(file, sep='\t', low_memory=False)
        if df.empty:
            print(f"警告：文件 {file} 为空，跳过")
        else:
            clinical_data.append(df)
            print(f"成功读取文件: {file}，样本数: {len(df)}，列名: {df.columns.tolist()}")
    except Exception as e:
        print(f"读取文件 {file} 时出错: {e}")

# 检查是否成功读取数据
if not clinical_data:
    print("错误：所有文件读取失败或为空，无法合并数据！")
    exit()

# 合并为一个 DataFrame
clinical_df = pd.concat(clinical_data, ignore_index=True)
print(f"合并完成，总样本数: {len(clinical_df)}")

# 处理生存时间和事件状态（适配不同的字段名）
def get_time_to_event(row):
    # 可能的死亡时间字段
    death_cols = ['days_to_death', 'death_days_to']
    # 可能的随访时间字段
    followup_cols = ['days_to_last_followup', 'days_to_last_follow_up', 'last_contact_days_to']
    
    # 获取死亡时间
    death_time = None
    for col in death_cols:
        if col in row.index and pd.notna(row[col]) and row[col] != '--':
            death_time = row[col]
            break
    
    # 获取随访时间
    followup_time = None
    for col in followup_cols:
        if col in row.index and pd.notna(row[col]) and row[col] != '--':
            followup_time = row[col]
            break
    
    # 根据 vital_status 判断
    if 'vital_status' in row.index:
        if row['vital_status'] == 'Dead' and death_time is not None:
            return death_time
        elif row['vital_status'] == 'Alive' and followup_time is not None:
            return followup_time
    
    # 如果 vital_status 缺失或无法判断，返回可用的时间
    if death_time is not None:
        return death_time
    if followup_time is not None:
        return followup_time
    return np.nan

clinical_df['time_to_event'] = clinical_df.apply(get_time_to_event, axis=1)
clinical_df['event'] = clinical_df['vital_status'].map({'Dead': 1, 'Alive': 0, '--': np.nan})

# 选择特征（动态调整）
features = [col for col in ['bcr_patient_barcode', 'age_at_diagnosis', 'gender', 'tumor_stage', 'time_to_event', 'event'] 
            if col in clinical_df.columns]
if not features:
    print("错误：未找到任何预期特征列！")
    print("可用列名：", clinical_df.columns.tolist())
    exit()

clinical_df = clinical_df[features].dropna(subset=['time_to_event', 'event'])
print(f"过滤后的样本数（移除缺失值）: {len(clinical_df)}")

# 编码分类变量（根据实际字段调整）
if 'gender' in clinical_df.columns:
    clinical_df['gender'] = clinical_df['gender'].map({'male': 0, 'female': 1, '--': 0}).fillna(0)
if 'tumor_stage' in clinical_df.columns:
    clinical_df['tumor_stage'] = clinical_df['tumor_stage'].map({
        'stage i': 1, 'stage ii': 2, 'stage iii': 3, 'stage iv': 4, 'not reported': 0, '--': 0
    }).fillna(0)
if 'age_at_diagnosis' in clinical_df.columns:
    clinical_df['age_at_diagnosis'] = clinical_df['age_at_diagnosis'].replace('--', np.nan).astype(float) / 365.25

# 保存为 CSV
output_file = 'tcga_clinical_processed.csv'
clinical_df.to_csv(output_file, index=False)
print(f"数据保存为 {output_file}")
print("样本数：", len(clinical_df))
print("最终列名：", clinical_df.columns.tolist())
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt
import joblib
import os
import sys

# 加载保存的模型
rsf_model = joblib.load('survival_analysis/rsf_model.pkl')

# 定义需要的特征及其说明
features_info = {
    'age': '患者年龄（单位：岁，例如 65，取值范围：0-120）',
    'gender_code': '性别（0=女性，1=男性，例如 1）',
    'tumor.size': '肿瘤大小（单位：毫米，例如 30.5，取值范围：0-1000）',
    'cea': '癌胚抗原水平（单位：ng/mL，例如 5.2，若未知输入 998 或 999）',
    'grade_code': '肿瘤分级（1=低级别，2=中级别，3=高级别，例如 2）',
    'PNI': '神经侵犯（0=无，1=有，例如 0）',
    'lymphcat': '淋巴结分类（0=无转移，1=有转移，例如 1）'
}

# 创建保存图像的目录
plot_dir = 'survival_analysis/predictions'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# 从命令行参数获取患者数据
if len(sys.argv) != 8:  # 期望 7 个参数 + 脚本名
    print("用法: python infer.py age gender_code tumor_size cea grade_code PNI lymphcat")
    sys.exit(1)

try:
    patient_data = {
        'age': float(sys.argv[1]),
        'gender_code': float(sys.argv[2]),
        'tumor.size': float(sys.argv[3]),
        'cea': float(sys.argv[4]),
        'grade_code': float(sys.argv[5]),
        'PNI': float(sys.argv[6]),
        'lymphcat': float(sys.argv[7])
    }

    # 范围检查
    if not (0 <= patient_data['age'] <= 120):
        print("年龄应在 0-120 岁之间")
        sys.exit(1)
    if patient_data['gender_code'] not in [0, 1]:
        print("性别只能是 0 或 1")
        sys.exit(1)
    if not (0 <= patient_data['tumor.size'] <= 1000):
        print("肿瘤大小应在 0-1000 毫米之间")
        sys.exit(1)
    if patient_data['grade_code'] not in [1, 2, 3]:
        print("肿瘤分级只能是 1、2 或 3")
        sys.exit(1)
    if patient_data['PNI'] not in [0, 1]:
        print("神经侵犯只能是 0 或 1")
        sys.exit(1)
    if patient_data['lymphcat'] not in [0, 1]:
        print("淋巴结分类只能是 0 或 1")
        sys.exit(1)

except ValueError:
    print("输入无效，请确保所有参数为数字")
    sys.exit(1)

# 将患者数据转换为 DataFrame
patient_df = pd.DataFrame([patient_data], columns=features_info.keys())

# 处理缺失值（如果医生输入 998/999，则替换为 NaN 并用训练集均值填充）
patient_df['cea'] = patient_df['cea'].replace([998, 999], np.nan)
if patient_df['cea'].isna().any():
    mean_cea = 5.0  # 请替换为实际训练集的均值
    patient_df['cea'].fillna(mean_cea, inplace=True)
    print(f"CEA 输入为缺失值，已填充为训练集均值 {mean_cea}")

# 使用 RSF 模型进行预测
X_patient = patient_df[list(features_info.keys())]

# 预测生存函数
surv_funcs = rsf_model.predict_survival_function(X_patient)
time_points = rsf_model.unique_times_

# 绘制生存曲线
plt.figure(figsize=(10, 6))
for i, surv_fn in enumerate(surv_funcs):
    plt.step(surv_fn.x, surv_fn.y, where="post", label=f"Patient {i+1}")
plt.title("Predicted Survival Curve for Patient")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'patient_survival_curve.png'))
plt.close()

# 预测风险分数
risk_score = rsf_model.predict(X_patient)[0]

# 使用训练集的风险中位数判断高/低风险（需从训练数据中获取，示例值）
median_risk = 50.0  # 请替换为实际训练集的中位数风险值
risk_group = "High Risk (建议积极治疗)" if risk_score > median_risk else "Low Risk (建议保守治疗)"

# 输出结果
print("\n预测结果：")
print(f"患者风险分数: {risk_score:.2f}")
print(f"风险分组: {risk_group}")
print(f"生存曲线已保存为: {os.path.join(plot_dir, 'patient_survival_curve.png')}")

# 输出生存概率的关键点
print("\n生存概率关键点：")
for t, prob in zip(surv_fn.x[::10], surv_fn.y[::10]):  # 每隔10个时间点输出一次
    print(f"时间 {t:.0f} 个月: {prob:.4f}")
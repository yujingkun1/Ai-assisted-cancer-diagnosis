import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import joblib

# 确保保存图像的目录存在
plot_dir = 'survival_analysis/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# 读取数据
df = pd.read_csv('survival_analysis/stage1_processed.csv')

# 数据预处理
print("原始样本数:", len(df))
print("缺失值统计：", df.isnull().sum())

# 处理 CEA 值（假设 998/999 为缺失值）
df['cea'] = df['cea'].replace([998, 999], np.nan)

# 动态选择特征
features = ['age', 'gender_code', 'tumor.size', 'cea', 'grade_code', 'PNI', 'lymphcat']
time_col = 'Survival.months'
event_col = 'status'

# 检查可用特征并过滤
available_features = [col for col in features if col in df.columns]
if not (time_col in df.columns and event_col in df.columns):
    print(f"错误：缺少 {time_col} 或 {event_col}")
    exit()

# 添加生存时间和事件状态到特征列表
df_model = df[available_features + [time_col, event_col]].dropna()
print(f"过滤后样本数: {len(df_model)}")

# 划分训练集和测试集
X = df_model[available_features]
y = Surv.from_dataframe(time=time_col, event=event_col, data=df_model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

# 训练随机生存森林
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
rsf.fit(X_train, y_train)

# 评估模型（使用 C-index）
print("\n评估模型...")
train_c_index = concordance_index_censored(y_train[event_col], y_train[time_col], rsf.predict(X_train))
test_c_index = concordance_index_censored(y_test[event_col], y_test[time_col], rsf.predict(X_test))
print(f"训练集 C-index: {train_c_index[0]:.4f}")
print(f"测试集 C-index: {test_c_index[0]:.4f}")

# 预测生存函数（示例：测试集前5个样本）
surv_funcs = rsf.predict_survival_function(X_test.iloc[:5])
plt.figure(figsize=(10, 6))
print("\n绘制生存函数...")
for i, surv_fn in enumerate(tqdm(surv_funcs, desc="Survival Functions")):
    plt.step(surv_fn.x, surv_fn.y, where="post", label=f"Sample {i}")
plt.title("Predicted Survival Functions (Random Survival Forest)")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.legend()
plt.savefig(os.path.join(plot_dir, 'survival_functions.png'))
plt.close()  # 关闭图像，防止显示

# 预测累积风险函数
chf_funcs = rsf.predict_cumulative_hazard_function(X_test.iloc[:5])
plt.figure(figsize=(10, 6))
print("\n绘制累积风险函数...")
for i, chf_fn in enumerate(tqdm(chf_funcs, desc="Cumulative Hazard Functions")):
    plt.step(chf_fn.x, chf_fn.y, where="post", label=f"Sample {i}")
plt.title("Predicted Cumulative Hazard Functions (Random Survival Forest)")
plt.xlabel("Time (Months)")
plt.ylabel("Cumulative Hazard")
plt.legend()
plt.savefig(os.path.join(plot_dir, 'cumulative_hazard_functions.png'))
plt.close()

# 计算测试集的风险分数并分组
risk_scores = rsf.predict(X_test)  # 预测风险分数
median_risk = np.median(risk_scores)  # 使用中位数划分高/低风险
high_risk = risk_scores > median_risk
low_risk = ~high_risk

# 绘制 Kaplan-Meier 曲线
plt.figure(figsize=(10, 6))
print("\n绘制 Kaplan-Meier 曲线...")
for group, label, color in [(high_risk, "High Risk (建议积极治疗)", "red"), (low_risk, "Low Risk (建议保守治疗)", "blue")]:
    time_km, survival_prob_km = kaplan_meier_estimator(y_test[event_col][group], y_test[time_col][group])
    plt.step(time_km, survival_prob_km, where="post", label=label, color=color)
plt.title("Kaplan-Meier Survival Curves by Risk Group")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.legend()
plt.savefig(os.path.join(plot_dir, 'kaplan_meier_curves.png'))
plt.close()

# 保存处理后的数据
df_model.to_csv('seer_processed_rsf.csv', index=False)
print("处理后的数据已保存为 seer_processed_rsf.csv")
print(f"图像已保存到 {plot_dir} 目录下")


# 保存 RSF 模型和标准化器（如果使用 Cox 模型则需要）
joblib.dump(rsf, 'survival_analysis/rsf_model.pkl')

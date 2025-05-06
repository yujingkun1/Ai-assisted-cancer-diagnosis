import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import joblib

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

# 标准化特征（Cox 模型需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练第一个 Cox 模型（之前的结果）
cox1 = CoxPHSurvivalAnalysis()
cox1.fit(X_train_scaled, y_train)

# 训练新的 Cox 模型（添加正则化参数 alpha=0.1 以区分）
cox2 = CoxPHSurvivalAnalysis(alpha=0.1)  # 添加 L2 正则化
cox2.fit(X_train_scaled, y_train)

# 训练 RSF 模型（之前的结果）
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
rsf.fit(X_train, y_train)  # RSF 不需要标准化

# 评估第一个 Cox 模型
print("\n评估 Cox1 模型...")
cox1_train_c_index = concordance_index_censored(y_train[event_col], y_train[time_col], cox1.predict(X_train_scaled))
cox1_test_c_index = concordance_index_censored(y_test[event_col], y_test[time_col], cox1.predict(X_test_scaled))
print(f"Cox1 训练集 C-index: {cox1_train_c_index[0]:.4f}")
print(f"Cox1 测试集 C-index: {cox1_test_c_index[0]:.4f}")

# 评估新的 Cox 模型
print("\n评估 Cox2 模型（带正则化）...")
cox2_train_c_index = concordance_index_censored(y_train[event_col], y_train[time_col], cox2.predict(X_train_scaled))
cox2_test_c_index = concordance_index_censored(y_test[event_col], y_test[time_col], cox2.predict(X_test_scaled))
print(f"Cox2 训练集 C-index: {cox2_train_c_index[0]:.4f}")
print(f"Cox2 测试集 C-index: {cox2_test_c_index[0]:.4f}")

# 评估 RSF 模型
print("\n评估 RSF 模型...")
rsf_train_c_index = concordance_index_censored(y_train[event_col], y_train[time_col], rsf.predict(X_train))
rsf_test_c_index = concordance_index_censored(y_test[event_col], y_test[time_col], rsf.predict(X_test))
print(f"RSF 训练集 C-index: {rsf_train_c_index[0]:.4f}")
print(f"RSF 测试集 C-index: {rsf_test_c_index[0]:.4f}")

# 模型对比
print("\n模型对比（测试集 C-index）：")
print(f"Cox1 测试集 C-index: {cox1_test_c_index[0]:.4f}")
print(f"Cox2 测试集 C-index: {cox2_test_c_index[0]:.4f}")
print(f"RSF 测试集 C-index: {rsf_test_c_index[0]:.4f}")

# 创建保存图像的目录
plot_dir = 'survival_analysis/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# 绘制并保存 Cox2 模型的 Kaplan-Meier 曲线
cox2_risk_scores = cox2.predict(X_test_scaled)
cox2_median_risk = np.median(cox2_risk_scores)
cox2_high_risk = cox2_risk_scores > cox2_median_risk
cox2_low_risk = ~cox2_high_risk

plt.figure(figsize=(10, 6))
print("\n绘制 Cox2 模型的 Kaplan-Meier 曲线...")
for group, label, color in [(cox2_high_risk, "Cox2 High Risk", "red"), (cox2_low_risk, "Cox2 Low Risk", "blue")]:
    time_km, survival_prob_km = kaplan_meier_estimator(y_test[event_col][group], y_test[time_col][group])
    plt.step(time_km, survival_prob_km, where="post", label=label, color=color)
plt.title("Kaplan-Meier Survival Curves by Cox2 Risk Group")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.legend()
plt.savefig(os.path.join(plot_dir, 'cox2_kaplan_meier.png'))
plt.close()  # 关闭图像，防止显示

# 绘制并保存 Cox1 模型的 Kaplan-Meier 曲线
cox1_risk_scores = cox1.predict(X_test_scaled)
cox1_median_risk = np.median(cox1_risk_scores)
cox1_high_risk = cox1_risk_scores > cox1_median_risk
cox1_low_risk = ~cox1_high_risk

plt.figure(figsize=(10, 6))
print("\n绘制 Cox1 模型的 Kaplan-Meier 曲线...")
for group, label, color in [(cox1_high_risk, "Cox1 High Risk", "red"), (cox1_low_risk, "Cox1 Low Risk", "blue")]:
    time_km, survival_prob_km = kaplan_meier_estimator(y_test[event_col][group], y_test[time_col][group])
    plt.step(time_km, survival_prob_km, where="post", label=label, color=color)
plt.title("Kaplan-Meier Survival Curves by Cox1 Risk Group")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.legend()
plt.savefig(os.path.join(plot_dir, 'cox1_kaplan_meier.png'))
plt.close()  # 关闭图像，防止显示

# 保存处理后的数据
df_model.to_csv('seer_processed_cox2_rsf.csv', index=False)
print("处理后的数据已保存为 seer_processed_cox2_rsf.csv")
joblib.dump(scaler, 'survival_analysis/scaler.pkl')  # 如果使用 Cox 模型需要保存
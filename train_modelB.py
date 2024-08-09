import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# 读取标注好的数据
data = pd.read_csv('labeled_sensor_data.csv')

# 将每100组数据作为一个样本，并计算每个样本的统计特征
num_samples = 100
X = []
y = []

for i in range(0, len(data), num_samples):
    sample = data.iloc[i:i+num_samples, :9].values
    if len(sample) == num_samples:
        # 计算统计特征
        means = np.mean(sample, axis=0)
        variances = np.var(sample, axis=0)
        medians = np.median(sample, axis=0)
        max_vals = np.max(sample, axis=0)
        min_vals = np.min(sample, axis=0)
        
        features = np.concatenate([means, variances, medians, max_vals, min_vals])
        
        label = data.iloc[i, 9]  # 假设第10列是标签
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 保存标准化器
joblib.dump(scaler, 'scaler.pkl')

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 评估模型
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 保存模型
joblib.dump(best_model, 'motion_model_optimized.pkl')

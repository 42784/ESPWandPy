import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# 加载模型和标准化器
model = joblib.load('motion_model_optimized.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['data']
    samples = data.split(';')
    
    sample_values = []

    for sample in samples:
        ax, ay, az, gx, gy, gz, pitch, roll, yaw = sample.split(',')
        values = [float(ax), float(ay), float(az), float(gx), float(gy), float(gz), float(pitch), float(roll), float(yaw)]
        sample_values.append(values)
    
    if len(sample_values) == 100:  # 检查是否有足够的数据
        sample_values = np.array(sample_values)
        
        # 计算统计特征
        means = np.mean(sample_values, axis=0)
        variances = np.var(sample_values, axis=0)
        medians = np.median(sample_values, axis=0)
        max_vals = np.max(sample_values, axis=0)
        min_vals = np.min(sample_values, axis=0)
        
        features = np.concatenate([means, variances, medians, max_vals, min_vals])
        
        # 标准化数据
        features = features.reshape(1, -1)
        features = scaler.transform(features)

        # 进行预测
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0]

        # 创建动作和相应概率的键值对
        label_encoder = joblib.load('label_encoder.pkl')
        actions = label_encoder.inverse_transform(np.arange(len(confidence)))
        result = {action: float(conf) for action, conf in zip(actions, confidence)}
        
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid data length'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

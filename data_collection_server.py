from flask import Flask, request
import csv

app = Flask(__name__)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    # 假设POST请求中的数据字段名为'data'
    data = request.form['data']
    # 假设数据以分号分隔，最后一个分号后是标签
    samples, label = data.rsplit(';', 1)
    samples = samples.split(';')
    
    # 存储所有样本的列表
    all_samples = []
    
    # 每100个样本作为一个数据点
    for i in range(0, len(samples), 100):
        sample_group = samples[i:i+100]
        # 确保我们有100个样本
        if len(sample_group) >= 100:
            # 将100个样本合并为一个样本
            for sample in sample_group:
                ax, ay, az, gx, gy, gz, pitch, roll, yaw = sample.split(',')
                all_samples.append([ax, ay, az, gx, gy, gz, pitch, roll, yaw, label])
    
    # 将所有样本写入CSV文件
    with open('labeled_sensor_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for sample in all_samples:
            writer.writerow(sample)
    
    return "Data received", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# ESPWandPy
## 这是魔杖的识别算法部分
### 使用的是随机森林算法
### 本人尝试过使用神经网络/深度学习 但是发现识别效果极差(可能是我数据量不够或者代码有问题)
###

## 数据收集 data_collection_server.py
### 数据发送到:5000/upload_data (可以修改ESP的代码直接发到这)
###

## 模型训练 train_modelB.py
### 直接运行即可
### 

## 识别 prediction_serverB.py
### POST数据到:5000/predict
### 结果返回对应动作的概率的键值对(可以修改直接返回最大值 在注释)
### 

import numpy as np
import pandas as pd
import onnx
import torch
import torch.onnx
import torch.nn as nn
from torch.onnx import utils
from sklearn.preprocessing import MinMaxScaler
import onnxruntime as ort

# 从 Excel 文件中读取数据
data_df = pd.read_excel(r"./data/20240628_0710.xlsx", sheet_name="Sheet1",engine='openpyxl')
num_files = data_df.shape[0]

# 选择需要的特征列和目标列
selected_features = ['AT', 'GHI']
target_column = 'pac'
# data = data_df[selected_features + [target_column]].values
dataX = data_df[selected_features].values
datay = data_df[[target_column]].values

time_data = pd.to_datetime(data_df['date'].astype(str) + ' ' + data_df['time'].astype(str))

data_time = time_data.values
#1    2019-08-01 06:33:35
#2    2019-08-01 06:33:42
#...

data_time = data_time.astype(str)
#['2019-08-01T06:33:29.000','2019-08-01T06:33:36.000',...]

# 归一化数据
# scaler1 = MinMaxScaler()
# data_normalized1 = scaler1.fit_transform(dataX)
#
# scaler2 = MinMaxScaler()
# data_normalized2 = scaler2.fit_transform(datay)

#使用Z-score归一化
dataX_mean = np.mean(dataX,axis=0)
dataX_std = np.mean(dataX,axis=0)
data_normalized1 = (dataX - dataX_mean) / dataX_std

datay_mean = np.mean(datay,axis=0)
datay_std = np.mean(datay,axis=0)
data_normalized2 = (datay - datay_mean) / datay_std

# 划分特征和目标变量
X = data_normalized1[:, :]  # 特征，shape为(50,3)
# y = data_normalized2[:, -1]  # 目标变量,shape为(100,)

# 组合特征与时间
X = np.concatenate((X, data_time.reshape(-1, 1)), axis=1) #原X的基础上加上时间列，X的shape变为(50,4)

# 划分训练集和测试集
X_test_ini = X
# y_test = y

X_test = X_test_ini[:, :2]
# X_test = X_test_ini[:, :3]      #取前3列，即 '辐照度', '环境温度','Pac'，此时X_test的shape为(50,3)
X_test = X_test.astype(np.float64)
X_test_time = X_test_ini[:, 2]
# X_test_time = X_test_ini[:, 3]  #取第3列，即时间列

torch.manual_seed(42)  # 保证每次跑的结果相同
# 转换为 PyTorch 张量

X_test = torch.tensor(X_test, dtype=torch.float32)
X_test = X_test.unsqueeze(1) #此时X_test的shape为(50,1,3)
X_test = X_test.unsqueeze(1) #此时X_test的shape为(50,1,1,3)

# 加载ONNX模型
sess = ort.InferenceSession("./model/lstm_2input_pac.onnx")

# 获取输入和输出名称
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

result = np.zeros((num_files, 1), dtype=np.float32)   # 创建推理结果张量
mean_error = np.arange(num_files, dtype=np.float32)   # 创建推理结果张量

# 运行ONNX模型
for i in range(0, num_files): #负数要等于0
    result[i] = np.array(sess.run([output_name], {input_name: X_test[i].numpy()}))
result = scaler2.inverse_transform(result) #把本来归一化的值进行逆转换。在这个过程中，MinMaxScaler会将result重新映射到原始数据的范围内，将标准化后的值恢复为原始值
result[result < 0] = 0
np.savetxt('./output/predict_2input_pac.txt', result, delimiter=',')

for i in range(0, num_files):
    mean_error[i] = abs((result[i] - datay[i])) / datay[i]
error_sum = np.sum(mean_error)
mean_error = error_sum / (len(mean_error))
print("The Mean Error = ", mean_error)

# 1）输入：辐照度、温度；输出：Qdc        mean error 0.36476478965930803
# 2）输入：辐照度、温度、Pac；输出：Qdc   mean error 0.31811596688620564
# 3）输入：辐照度、温度；输出：Pdc        mean error 0.3737437809997154
# 4）输入：辐照度、温度、Pac；输出：Pdc   mean error 0.12675356433203236

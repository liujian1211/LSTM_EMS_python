from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
import torch

#拟合的历史数据
data = pd.read_excel('/home/ai/liujian/LSTM_RK3568/python_code/data/20240708_1645_1745.xlsx')

#待预测的数据
data_predict = pd.read_excel('/home/ai/liujian/LSTM_RK3568/python_code/data/20240708_1745_1750.xlsx')

# 设置GHI和AT的拟合相关的参数
p_ghi=0
d_ghi=2
q_ghi=2

p_at=0
d_at=1
q_at=0

at_order = (p_at,d_at,q_at)
ghi_order = (p_ghi,d_ghi,q_ghi)

#拟合算法
def fit_arima_model(series,order):
    model = ARIMA(series,order=order)
    result = model.fit()
    return result

time_series_ghi = data.set_index('time')['GHI']  # 取GHI作为时间序列数据
time_series_at = data.set_index('time')['AT']    # 取AT作为时间序列数据

time = data['time']
GHI = data['GHI']
AT = data['AT']

GHI_future = data_predict['GHI']
AT_future  = data_predict['AT']

# 拟合模型
temperature_model = fit_arima_model(time_series_at, at_order)
ghi_model = fit_arima_model(time_series_ghi, ghi_order)


time_data = pd.to_datetime(data_predict['date'].astype(str) + ' ' + data_predict['time'].astype(str))
date_time = time_data.values
date_time = date_time.astype(str)

#预测5min，确保预测长度和待预测时间一致
predictions_at = temperature_model.predict(start=len(AT),end=len(AT)+len(date_time)- 1,dynamic=True)
predictions_ghi = ghi_model.predict(start=len(GHI),end = len(AT)+len(date_time)- 1,dynamic=True)

dataX = np.column_stack((predictions_at.values,predictions_ghi.values))
dataY = data[['pac']].values

scaler1 = MinMaxScaler()
data_normalized1 = scaler1.fit_transform(dataX)

scaler2 = MinMaxScaler()
scaler2.fit_transform(dataY) #根据前1小时的数据拟合未来的结果


X = data_normalized1[:, :]
X = np.concatenate((X, date_time.reshape(-1, 1)), axis=1) #原X的基础上加上时间列，X的shape变为(50,4)

X_test_ini = X
X_test = X_test_ini[:, :2]
X_test = X_test.astype(np.float64)

torch.manual_seed(42)

X_test = torch.tensor(X_test, dtype=torch.float32)
X_test = X_test.unsqueeze(1)
X_test = X_test.unsqueeze(1)

#-----------------预测部分------------------------
num_files = data_predict.shape[0]  #待预测的数量

# 加载ONNX模型
sess = ort.InferenceSession("./model/lstm_2input_pac.onnx")

# 获取输入和输出名称
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

result = np.zeros((num_files, 1), dtype=np.float32)   # 创建推理结果张量
mean_error = np.arange(num_files, dtype=np.float32)   # 创建推理结果张量

#运行onnx模型
for i in range(0,num_files):
    result[i] = np.array(sess.run([output_name],{input_name:X_test[i].numpy()}))
result = scaler2.inverse_transform(result) #把本来归一化的值进行逆转换。在这个过程中，MinMaxScaler会将result重新映射到原始数据的范围内，将标准化后的值恢复为原始值
result[result < 0] = 0
np.savetxt('./output/predict_1745_1750.txt', result, delimiter=',')

print(f'预测的发电功率的均值为{np.mean(result)}')

# print(f'实际GHI和AT的均值为{np.mean(GHI_future.values)}和{np.mean(AT_future.values)}')
# print(f'预测GHI和AT的均值为{np.mean(predictions_ghi.values)}和{np.mean(predictions_at.values)}')

# plt.plot(range(63, len(GHI_future) + 63),GHI_future,label='real GHI')
# plt.plot(range(63, len(GHI_future) + 63),AT_future,label='real AT')
#
# plt.plot(predictions_ghi,label='predicted GHI')
# plt.plot(predictions_at,label='predicted AT')
# plt.xticks(range(0, 100, 5))  # 设置x轴刻度从到100，间隔为10
# plt.legend()
# plt.show()

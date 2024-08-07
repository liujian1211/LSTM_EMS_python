import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 从 Excel 文件中读取数据
data_df = pd.read_excel(r"E:\project\PV-generation-forecasting\【2019HWJC001】项目2\2019.08.xlsx",
                        sheet_name="#18-2")
# # 使用query函数进行筛选（根据'bsrn_pass'字段进行筛选，为0删除，为1保留）
# data_df = data_df.query('bsrn_pass == 1')
# data_df = data_df[data_df['Measured DC power (W)'] > 100]  # <class 'pandas.core.frame.DataFrame'>

# 选择需要的特征列和目标列
selected_features = ['辐照度', '环境温度']
target_column = 'Qdc'
# data = data_df[selected_features + [target_column]].values
dataX = data_df[selected_features].values
datay = data_df[[target_column]].values
# time_data = pd.to_datetime(data_df[["Year", "Month", "Day", "Hour"]])  # 时间列（合并）
time_data = pd.to_datetime(data_df['日期'].astype(str) + ' ' + data_df['时间'].astype(str))
# print(type(time_data))
# print(time_data)
data_time = time_data.values
data_time = data_time.astype(str)

# 归一化数据
scaler = MinMaxScaler()
data_normalized1 = scaler.fit_transform(dataX)
data_normalized2 = scaler.fit_transform(datay)
# scaler2 = MinMaxScaler()
# data_normalized2 = scaler.fit_transform(data_df[target_column].values)

# 划分特征和目标变量
X = data_normalized1[:, :]  # 特征
y = data_normalized2[:, -1]  # 目标变量
# 组合特征与时间
X = np.concatenate((X, data_time.reshape(-1, 1)), axis=1)


# 划分训练集和测试集
# X_train_ini, X_test_ini, y_train, y_test = train_test_split(X, y, test_size=2, random_state=42)
X_test_ini = X
y_test = y
# X_train = X_train_ini[:, :2]
# X_train = X_train.astype(np.float64)
# X_train_time = X_train_ini[:, 2]
X_test = X_test_ini[:, :2]
X_test = X_test.astype(np.float64)
X_test_time = X_test_ini[:, 2]
# print(X_test_time)
torch.manual_seed(42)  # 保证每次跑的结果相同
# 转换为 PyTorch 张量
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_train = X_train.unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_test = X_test.unsqueeze(1)
# y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# print(X_test)

# 加入注意力机制
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(AttentionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size, 1)  # 注意力权重计算
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h_0, c_0))
        attention_scores = self.attention(out).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_out = torch.bmm(attention_weights.unsqueeze(1), out).squeeze(1)

        final_output = self.fc(attention_out)
        return final_output

    def save(self, path):
        # 保存模型参数
        torch.save(self.state_dict(), path)

    # @staticmethod
    # def load(path):
    #     # 加载模型参数
    #     model = AttentionLSTM(len(selected_features), 256, 10, 1)  # 注意！！！！！！跟直接预测文件input_size, hidden_size,
    #     # num_layers, output_size四个参数保持一致
    #     model.load_state_dict(torch.load(path))
    #     return model

    # def load(path):
    #     # 加载HDF5文件
    #     with h5py.File(path, 'r') as f:
    #         # 创建一个空字典来保存模型参数
    #         state_dict = {}
    #
    #         # 遍历HDF5文件中的数据集，将参数加载到字典中
    #         for name, param in f.items():
    #             state_dict[name] = torch.tensor(param[:])



# # 定义 LSTM 模型
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
#         super(LSTMModel, self).__init__()
#
#         # 设置超参数
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         # 添加模型层
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)  # LSTM层
#         self.fc = nn.Linear(hidden_size, output_size)  # 全连接层
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         out = self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出
#         return out


# 初始化模型
input_size = len(selected_features)  # 输入特征数
hidden_size = 128  # 隐藏层256，单独lstm128
num_layers = 3  # LSTM层8，单独lstm10
output_size = 1
dropout = 0.0  # dropout层，防止过拟合
mlp_hidden_size = 256
bidirectional = False  # 是否使用双向LSTM



# 加载模型
loaded_model = AttentionLSTM(input_size, hidden_size, num_layers, output_size)
loaded_model.load_state_dict(torch.load('pv_power_predict_model_4_10.pth'))



# 在测试集上进行预测
# loaded_model.eval()
# with torch.no_grad():
#     y_pred = load_model(X_test)
y_pred1 = loaded_model(X_test)
tensor_with_grad = torch.tensor(y_pred1, requires_grad=True)
y_pred = tensor_with_grad.detach().clone()
y_pred = y_pred.squeeze(dim=0)

# y_pred = loaded_model(y_pred).detach().numpy()
# print(y_pred)
# print(type(y_pred))
# print(y_test)
# print(type(y_test))

# 反归一化预测结果
y_pred_unscaled = scaler.inverse_transform(y_pred.numpy())
print(type(y_pred_unscaled))
print('预测值：', y_pred_unscaled)

# 反归一化真实值
y_test_unscaled = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
print('测试值：', y_test_unscaled)

# 计算相对误差
relative_error = np.abs(y_pred_unscaled - y_test_unscaled) / np.abs(y_test_unscaled)
mean_relative_error = np.mean(relative_error)
print("平均相对误差%：", mean_relative_error * 100)

# # 删除高误差数据
# relative_error_edit = []  # 存放筛选后的相对误差
# y_test_unscaled_edit = []  # 存放筛选后的测试集
# y_pred_unscaled_edit = []  # 存放筛选后的预测集
# time_unscaled_edit = []  # 存放筛选后的时间
# for i in range(len(relative_error)):
#     if relative_error[i] <= 0.7:  # 前两次0.7
#         relative_error_edit.append(relative_error[i])
#         y_test_unscaled_edit.append(y_test_unscaled[i])
#         y_pred_unscaled_edit.append(y_pred_unscaled[i])
#         time_unscaled_edit.append(X_test_time[i])
# mape = np.mean(relative_error_edit) * 100
# print('修改后平均相对误差%', mape)















"""均方误差图"""
# # 计算预测值与真实值的均方误差
# mse = mean_squared_error(y_test_unscaled_edit, y_pred_unscaled_edit)
# # 绘制散点图和趋势线
# plt.scatter(y_test_unscaled_edit, y_pred_unscaled_edit)
# plt.plot([min(y_test_unscaled_edit), max(y_test_unscaled_edit)], [min(y_test_unscaled_edit), max(y_test_unscaled_edit)], linestyle='--', color='red')
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.title(f'Mean Squared Error: {mse:.2f}')
# plt.show()
#
# def mean_squared_error(y_true, y_pred):
#     n = len(y_true)
#     for i in range(len(y_true)):
#         mse1 = np.sum((y_true[i] - y_pred[i]) ** 2) / n
#         print(mse1)
#
# mean_squared_error(y_test_unscaled_edit, y_pred_unscaled_edit)
# def mean_squared_error(y_true, y_pred):
#     mse = []
#     n = len(y_true)
#     for i in range(len(y_true)):
#         mse.append((y_true[i] - y_pred[i]) ** 2)
#     print(mse)
# mean_squared_error(y_test_unscaled_edit, y_pred_unscaled_edit)

# """均绝对误差MAE"""
# mae = mean_absolute_error(y_test_unscaled_edit, y_pred_unscaled_edit)
# print('MAE:', mae)
# # 绘制散点图和趋势线
# plt.scatter(y_test_unscaled_edit, y_pred_unscaled_edit, s=3)
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 字体设置
# plt.plot([min(y_test_unscaled_edit), max(y_test_unscaled_edit)], [min(y_test_unscaled_edit), max(y_test_unscaled_edit)], linestyle='--', color='red')
# plt.xlabel("Actual Power Generation(W)")
# plt.ylabel("Predicted Power Generation(W)")
# plt.title(f'Mean Absolute Error(MAE): {mae:.2f}')  # {mae:.2f}
# plt.show()
#
# """平均绝对百分比误差MAPE"""
# # 绘制MAPE图表
# plt.figure(figsize=(8, 6))
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 字体设置
# plt.plot(y_test_unscaled_edit[:100], label='Actual Value', linestyle='--', marker='x', markersize=3)
# plt.plot(y_pred_unscaled_edit[:100], label='Predicted Value', linestyle='--', marker='x', markersize=3)
# plt.xlabel('Sample Size')
# plt.ylabel('Power Generation(W)')
# plt.title(f'Mean Absolute Percentage Error(MAPE): {mape:.2f}')  # {mape:.2f}
# plt.legend()
# plt.show()
#
# """均方根误差RMSE"""
# # 计算均方根误差
# rmse = np.sqrt(np.mean(np.square(np.array(y_test_unscaled_edit) - np.array(y_pred_unscaled_edit))))
# print('RMSE:', rmse)
# # 绘制均方根误差图
# plt.figure(figsize=(8, 6))
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 字体设置
# plt.plot(y_test_unscaled_edit[:100], linestyle='--', marker='x', label='Actual Value')
# plt.plot(y_pred_unscaled_edit[:100], linestyle='--', marker='x', label='Predicted Value')
# plt.legend()
# plt.title(f'Root Mean Squared Error(RMSE): {rmse:.2f}')  # {rmse:.2f}
# plt.xlabel('Sample Size')
# plt.ylabel('Power Generation(W)')
# plt.show()


"""典型日预测结果"""
# 预处理输入数据
# input_data = [[0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0],
#               [21.56, 0.13, 21.70, 92.67, 1.41, 22.16, 17.14],
#               [100.71, 85.82, 76.81, 84.22, 1.93, 81.06, 18.36],
#               [297.76, 184.60, 240.61, 64.73, 1.65, 249.48, 27.66],
#               [569.97, 606.13, 191.93, 56.68, 1.86, 493.55, 36.51],
#               [768.87, 905.22, 72.09, 50.77, 2.24, 725.75, 46.14],
#               [903.90, 926.89, 84.77, 46.19, 3.10, 884.37, 51.18],
#               [981.25, 950.85, 83.07, 43.71, 3.14, 972.26, 55.44],
#               [1007.95, 907.88, 136.94, 41.85, 3.16, 1011.07, 57.14],
#               [510.25, 378.01, 153.00, 37.40, 3.13, 501.50, 46.55],
#               [79.18, 0.57, 71.57, 63.45, 6.31, 74.68, 21.00],
#               [50.51, 0.19, 51.88, 78.94, 3.55, 51.52, 18.98],
#               [143.77, 17.56, 142.48, 69.09, 2.70, 126.28, 23.87],
#               [129.19, 126.12, 89.46, 61.73, 3.41, 93.29, 24.08],
#               [23.15, 0.26, 23.14, 63.81, 3.59, 18.66, 20.76],
#               [0.82, 0.10, 1.63, 63.05, 3.87, 3.73, 20.75],
#               [0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0]]  # 需要预测的数据
# y_data = [[0],
#           [0],
#           [0],
#           [0],
#           [0],
#           [64.91],
#           [243.91],
#           [831.93],
#           [1716.14],
#           [2528.49],
#           [3048.94],
#           [3304.51],
#           [3446.05],
#           [1742.16],
#           [272.38],
#           [177.13],
#           [426.03],
#           [282.28],
#           [52.15],
#           [3.34],
#           [0],
#           [0],
#           [0],
#           [0]]
# input_data = np.array(input_data)
# y_data = np.array(y_data)
#
# data_normalized3 = scaler.fit_transform(input_data)
# input_data = data_normalized3[:, :]
# data_normalized4 = scaler.fit_transform(y_data)
# y_data = data_normalized4[:, -1]
#
# input_tensor = torch.tensor(input_data, dtype=torch.float32)
# input_tensor = input_tensor.unsqueeze(1)
#
# y_tensor = torch.tensor(y_data, dtype=torch.float32)
#
#
# # 将数据输入模型进行预测
# with torch.no_grad():
#     prediction = model(input_tensor)
#
# # 解码预测结果（示例为回归任务）
# predicted_values = scaler.inverse_transform(prediction.squeeze().numpy().reshape(-1, 1))
# predicted_values[predicted_values < 0] = 0
# print(predicted_values)
# y_tensor = scaler.inverse_transform(y_tensor.numpy().reshape(-1, 1))
# print(y_tensor)
#
#
#
# # # 反归一化预测结果
# # y_pred_unscaled = scaler.inverse_transform(y_pred.numpy())
# # # print(y_pred_unscaled)
# #
# # # 反归一化真实值
# # y_test_unscaled = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
#
#
#
#
# # 绘制预测结果图表
# plt.figure(figsize=(10, 6))
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 字体设置
# plt.plot(y_tensor, label='原始数据')
# plt.plot(predicted_values, label='预测结果', linestyle='dashed')
# plt.xlabel('时间步')
# plt.ylabel('预测值')
# plt.title('LSTM模型预测结果')
# plt.legend()
# plt.grid(True)
# plt.show()

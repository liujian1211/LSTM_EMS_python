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

# 选择需要的特征列和目标列
selected_features = ['AT', 'GHI']
target_column = 'pac'

dataX = data_df[selected_features].values
datay = data_df[[target_column]].values

time_data = pd.to_datetime(data_df['date'].astype(str) + ' ' + data_df['time'].astype(str))


data_time = time_data.values
#data_time:
#2019-08-01T06:33:29.000
#2019-08-01T06:33:36.000
#2019-08-01T06:33:44.000
#...一共131791行

data_time = data_time.astype(str)  #将数组或者张量的数据类型转换为字符串类型
#data_time:上述的data_time变为数组，shape为(131791,)
#['2019-08-01T06:33:29.000','2019-08-01T06:33:36.000',...]

# 归一化数据
scaler1 = MinMaxScaler()
data_normalized1 = scaler1.fit_transform(dataX)

scaler2 = MinMaxScaler()
data_normalized2 = scaler2.fit_transform(datay)


# 划分特征和目标变量
X = data_normalized1[:, :]  # 特征，shape为(131791,2),nd_array格式
y = data_normalized2[:, -1]  # 目标变量，data_normalized2的原本的shape为(131791,1)，现只取最后一列，shape变为了(131791,)

# 组合特征与时间,data_time.reshape(-1, 1)后的shape为(131791, 1)
X = np.concatenate((X, data_time.reshape(-1, 1)), axis=1) #在axis=1，即列的方向 组合X和data_time.reshape(-1, 1)
# print(f'X的shape为{X.shape}')  #应该是(131791,3)

# 划分训练集和测试集
X_test_ini = X
y_test = y

X_test = X_test_ini[:, :5]  #取所有的行，前2列，即不包含时间的那2列
# X_test = X_test_ini[:, :3]  #取所有的行，前3列，即不包含时间的那2列
X_test = X_test.astype(np.float64)  #字符串转为float64
X_test_time = X_test_ini[:, 5]  #取所有的行，第2列，即时间列
# X_test_time = X_test_ini[:, 3]  #取所有的行，第2列，即时间列

torch.manual_seed(42)  # 保证每次跑的结果相同
# 转换为 PyTorch 张量

X_test = torch.tensor(X_test, dtype=torch.float32)  #转为张量
X_test = X_test.unsqueeze(1)        #shape由(131791,2)转为(131791,1,2)  #shape由(131791,3)转为(131791,1,3)

y_test = torch.tensor(y_test, dtype=torch.float32)
y_test = y_test.unsqueeze(1)   #shape由(131791,)转为(131791,1)

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
        batch_size, seq_len, _ = x.size()  #(131791,1,2)  #(131791,1,3)

        #LSTM的初始隐藏状态和细胞状态，通常是0
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)    #shape为(3, 131791, 128)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)    #shape为(3, 131791, 128)

        out, _ = self.lstm(x, (h_0, c_0)) #out的shape为(1,131791,128)，即(seq_len, batch,hidden_size)
        #lstm返回2个值，output和(h_n,c_n)，其中
        #output: 这个是LSTM对输入序列的每个时间步的输出。它的shape是 (seq_len, batch, num_directions * hidden_size)，其中：
        # seq_len 是输入序列的长度。
        # batch 是批次大小。
        # num_directions 是方向的数量，1 表示单向LSTM，2 表示双向LSTM。
        # hidden_size 是每个LSTM单元的隐藏状态的大小。

        #(h_n, c_n): 这是最后一个时间步的隐藏状态和细胞状态。每个状态都是一个元组，包括两个张量：
        # h_n：形状为 (num_layers * num_directions, batch, hidden_size)，表示最后一个时间步的隐藏状态。
        # c_n：形状为 (num_layers * num_directions, batch, hidden_size)，表示最后一个时间步的细胞状态。

        attention_scores = self.attention(out).squeeze(2)  #self.attention(out)后的shape为(1, 131791, 1)，squeeze后的shape为(1, 131791)
        attention_weights = torch.softmax(attention_scores, dim=1) #shape仍为(1, 131791)，每个元素归一化了
        attention_out = torch.bmm(attention_weights.unsqueeze(1), out).squeeze(1)
        # attention_weights.unsqueeze(1)的shape为(1,1,131791)，out的shape为(1,131791,128)，使用torch.bmm对两者使用批量矩阵乘法，得到(1,1,128)的数据，squeeze(1)后得到(1,128)的数据

        final_output = self.fc(attention_out)  #得到一个(1,1)的值
        return final_output

    def save(self, path):
        # 保存模型参数
        torch.save(self.state_dict(), path)

# 初始化模型
input_size = len(selected_features)  # 输入特征数
hidden_size = 128  # 隐藏层256，单独lstm128
num_layers = 3  # LSTM层8，单独lstm10
output_size = 1  #输出维度
dropout = 0.0  # dropout层，防止过拟合
mlp_hidden_size = 256
bidirectional = False  # 是否使用双向LSTM

# 加载模型
loaded_model = AttentionLSTM(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):

    # 前向传播
    outputs = loaded_model(X_test)  #X_test此时shape为(131791,1,2)
    loss = criterion(outputs, y_test)  #Y_test此时shape位(131791,1)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失值
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


with torch.no_grad():
    y_pred = loaded_model(X_test)
    mse = criterion(y_pred, y_test)
    print(f'Mean Squared Error on test data: {mse.item():.4f}')

# torch.onnx.export(loaded_model, X_test, "lstm_model.onnx", input_names=["input"], output_names=["output"])

# # 设置模型为评估模式
loaded_model.eval()

batchSize = 1

# 定义输入张量的大小
input_shape = (1, 1, len(selected_features))

# 创建一个模拟输入张量
dummy_input = torch.randn(input_shape)

# 指定输出文件名
output_file = "./model/lstm_2input_pac.onnx"

# 将模型导出为ONNX格式
torch.onnx.export(loaded_model, dummy_input, output_file, 
input_names = ["input"],
output_names = ["output"],
opset_version=12)

# # 加载 ONNX 模型
# model = onnx.load("./model/lstm_v0.onnx")

# # 将 ONNX 模型转换为 PyTorch 模型
# torch_model = torch.onnx.export(model, "./model/lstm_v0.pth", X_test)
# loaded_model.load_state_dict(torch.load('./model/lstm_v0.onnx'))

# # 在测试集上进行预测
# y_pred1 = loaded_model(X_test)
# tensor_with_grad = torch.tensor(y_pred1, requires_grad=True)
# y_pred = tensor_with_grad.detach().clone()
# y_pred = y_pred.squeeze(dim=0)

# # 反归一化预测结果
# y_pred_unscaled = scaler.inverse_transform(y_pred.numpy())
# print(type(y_pred_unscaled))
# print('预测值：', y_pred_unscaled)

# # 反归一化真实值
# y_test_unscaled = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
# print('测试值：', y_test_unscaled)

# # 计算相对误差
# relative_error = np.abs(y_pred_unscaled - y_test_unscaled) / np.abs(y_test_unscaled)
# mean_relative_error = np.mean(relative_error)
# print("平均相对误差%：", mean_relative_error * 100)

# 加载ONNX模型
# sess = ort.InferenceSession("lstm_model.onnx")

# # 获取输入和输出名称
# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name

# # 运行ONNX模型
# result = sess.run([output_name], {input_name: X_test.numpy()})
# print("ONNX模型预测结果：", result)
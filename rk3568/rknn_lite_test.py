import numpy as np
from rknnlite.api import RKNNLite
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler

# 设置rknn模型和测试数据文件夹的路径
# RKNN_MODEL = './model/lstm.rknn'
RKNN_MODEL = './model/lstm_2input_Pdc_.rknn'

input_expand = 1            # 与输入保持一致，扩展一个维度
input_high = 1              # 由于测试数据为一维，所以将数据的另一个维度设置为1
input_width = 2           # 输入数据的维度

# 从 Excel 文件中读取数据
data_df = pd.read_excel(r"./data/test_data.xlsx", sheet_name="#18-2",engine='openpyxl')
num_files = data_df.shape[0]

# 选择需要的特征列和目标列
selected_features = ['辐照度', '环境温度']
target_column = 'Pdc'

dataX = data_df[selected_features].values
datay = data_df[[target_column]].values

# 归一化数据
scaler1 = MinMaxScaler()
data_normalized1 = scaler1.fit_transform(dataX)

scaler2 = MinMaxScaler()
data_normalized2 = scaler2.fit_transform(datay)

# 获取归一化数据的最大值和最小值
print("DataX_Min = ", scaler1.data_min_)
print("DataX_Max = ", scaler1.data_max_)
print("DataY_Min = ", scaler2.data_min_)
print("DataY_Max = ", scaler2.data_max_)

# 导入推理测试数据
input_data = data_normalized1.reshape((num_files, input_expand, input_high, input_width))
input_data = input_data.astype(np.float32)

# 初始化变量
ans = np.zeros((num_files, input_expand), dtype=np.float32)   # 创建推理结果张量
mean_error = np.arange(num_files*input_expand, dtype=np.float32)   # 创建推理结果张量

# 加速推理的主程序
if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNNLite()

    # Load model
    print('--> Loading model')
    ret = rknn.load_rknn(path=RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    for i in range(0, num_files):
        ans[i] = np.array(rknn.inference(inputs=[input_data[i]]))
    ans = scaler2.inverse_transform(ans)
    np.savetxt('./output/output_Pdc.csv', ans, delimiter=',')

    for i in range(0, num_files):
        mean_error[i] = abs((ans[i, 0] - datay[i])) / datay[i]
    error_sum = np.sum(mean_error)
    mean_error = error_sum / (len(mean_error))
    print("The Mean Error = ", mean_error)

    # 将数据写入CSV文件
    with open('MaxMin.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer1 = csv.writer(csvfile)
        for row in data_normalized1:
            csv_writer1.writerow(row)

    with open('Predict.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer2 = csv.writer(csvfile)
        for row2 in ans:
            csv_writer2.writerow(row2)

    rknn.release()

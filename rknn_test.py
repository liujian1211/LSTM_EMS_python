import numpy as np
from rknn.api import RKNN
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 设置rknn模型和测试数据文件夹的路径
RKNN_MODEL = './rk3568/model/lstm_2input_Pdc.rknn'

input_expand = 1            # 与输入保持一致，扩展一个维度
input_high = 1              # 由于测试数据为一维，所以将数据的另一个维度设置为1
input_width = 2           # 输入数据的维度

# 从 Excel 文件中读取数据
data_df = pd.read_excel(r"./data/test_data_201909.xlsx", sheet_name="#18-2",engine='openpyxl', nrows=100)
predict = np.loadtxt('./output/predict.csv', delimiter=',', skiprows=0, dtype=np.float32)

num_files = data_df.shape[0]

# 选择需要的特征列和目标列
selected_features = ['辐照度', '环境温度']
#target_column = 'Qdc'  #发电量
target_column = 'Pdc'  #发电量

dataX = data_df[selected_features].values
datay = data_df[[target_column]].values
# print("datay = ", datay)

# 归一化数据
scaler1 = MinMaxScaler()
data_normalized1 = scaler1.fit_transform(dataX)

scaler2 = MinMaxScaler()
data_normalized2 = scaler2.fit_transform(datay)

# 导入推理测试数据
input_data = data_normalized1.reshape((num_files, input_expand, input_high, input_width))
input_data = input_data.astype(np.float32)

np.savetxt('./output/input.csv', input_data.reshape((num_files, input_width)), delimiter=',')
# print('true_data = ',input_data.shape)
np.savetxt('./output/output_raw.csv', datay, delimiter=',')
# print('output_size = ',y_test.shape)

# 初始化变量
ans = np.zeros((num_files, input_expand), dtype=np.float32)   # 创建推理结果张量
mean_error1 = np.arange(num_files*input_expand, dtype=np.float32)   # 创建推理结果张量
mean_error2 = np.arange(num_files*input_expand, dtype=np.float32)   # 创建推理结果张量

# 加速推理的主程序
if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Load model
    print('--> Loading model')
    ret = rknn.load_rknn(path=RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(
        target=None,
        device_id=None
    )
    
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    for i in range(0, num_files):
        ans[i] = rknn.inference(inputs=[input_data[i]])
        # print("input_data = ", input_data[i])
        # print("predict = ", outputs)
        # print("true_data = ", predict[i])
    ans = scaler2.inverse_transform(ans)

    # print("input_size = ", input_data[i].shape)
    # print("ans_size = ",ans.shape)
    # print('output_size = ',true_data.shape)
    np.savetxt('./output/output.csv', ans, delimiter=',')
    error_sum = 0
    for i in range(0, num_files):
        error_sum = error_sum + abs((ans[i, 0] - datay[i])) / datay[i]
    mean_error1 = error_sum / num_files
    print("与真值误差为：", mean_error1)

    error_sum1 = 0
    for i in range(0, num_files):
        error_sum1 = error_sum1 + abs((ans[i, 0] - predict[i])) / predict[i]
    mean_error2 = error_sum1 / num_files
    print("与预测值误差为：", mean_error2)

    rknn.release()

import os
import numpy as np
from rknn.api import RKNN
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 设置rknn模型和测试数据文件夹的路径
ONNX_MODEL = './model/lstm_2input_pdc_shift_xlp.onnx'
QUANT_DATA = './dataset_2input_pdc_shift_xlp.txt'
RKNN_MODEL = './rk3568/model/lstm_2input_pdc_shift_xlp.rknn'

input_expand = 1            # 与输入保持一致，扩展一个维度
input_high = 1              # 由于测试数据为一维，所以将数据的另一个维度设置为1
input_width = 2             # 输入数据的维度

# 从 Excel 文件中读取数据
data_df = pd.read_excel(r"./data/0840_0900_xlp.xlsx", sheet_name="Sheet1",engine='openpyxl', nrows=1)  #读取第一行，不包括表头
#data_df的shape应该为(1,13)

# num_files = data_df.shape[0]
#
# print(num_files)

# 选择需要的特征列和目标列
selected_features = ['辐照度', '环境温度']
# selected_features = ['AH','AT', 'GHI','GTI','MT']
# target_column = 'Pdc'
dataX = data_df[selected_features].values

# 归一化数据
scaler1 = MinMaxScaler()
data_normalized1 = scaler1.fit_transform(dataX)

#使用Z-score归一化
# dataX_mean = np.mean(dataX,axis=0)
# dataX_std = np.mean(dataX,axis=0)
# data_normalized1 = (dataX - dataX_mean) / dataX_std

# 导入推理测试数据
input_data = data_normalized1.reshape((input_expand, input_high, input_width))
input_data = input_data.astype(np.float32)

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')

    rknn.config(target_platform='rk3568')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
        input_size_list=[[input_expand, input_high, input_width]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(
        do_quantization=True,
        dataset=QUANT_DATA)

    if ret != 0:
        print('Build model failed!')
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
    
    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # # 量化精度分析
    rknn.accuracy_analysis(
        inputs = [input_data],
        output_dir = "./output", # 表示精度分析的输出目录
        target=None,
        device_id=None
    )

    rknn.release()

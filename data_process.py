#该文件在raw_data.xlsx数据中取前50个数据；
#将上述数据归一化，每个数据的shape为(1,1,2)，例如[[[0. 0.]]]
#上述每个数据作为的值写入一个numpy文件中，即test0.npy-test49.npy
#上述每个npy文件的路径写进dataset.txt中

import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import MinMaxScaler

# 从 原始Excel 文件中读取数据
data_df = pd.read_excel(r"./data/2019_08_min.xlsx", sheet_name="Sheet1",engine='openpyxl', nrows=50)
#data_df的shape应该为(50,13)

num_files = data_df.shape[0]

# 选择需要的特征列和目标列
selected_features = ['辐照度', '环境温度']
# selected_features = ['AH','AT', 'GHI','GTI','MT']
dataX = data_df[selected_features].values

# print(dataX)

# 归一化数据
scaler = MinMaxScaler()
data_normalized1 = scaler.fit_transform(dataX)

# 转换为 PyTorch 张量
X_test = torch.tensor(data_normalized1, dtype=torch.float32) #此时X_test的shape为(50,2)
X_test = X_test.unsqueeze(1)                                 #在第1纬度新增1个纬度，此时X_test的shape为(50,1,2)
X_test = X_test.unsqueeze(1)                                 #在第1纬度新增1个纬度，此时X_test的shape为(50,1,1,2)
# print("input_size = ", X_test.shape)

def delete_files(directory):
    file_list = os.listdir(directory)
    for file in file_list:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

DATASET_PATH = './dataset_2input_pdc_shift_xlp.txt'
if os.path.exists(DATASET_PATH):
    os.remove(DATASET_PATH)

QUANT_PATH = './data/quant'
delete_files(QUANT_PATH)  #删掉'./data/quant'里的所有npy文件

file = open(DATASET_PATH, 'a')  #a 模式：以追加模式打开文件。如果该文件已经存在，将会在文件末尾追加写入内容，而不会覆盖已有内容。如果文件不存在，将会创建一个新的文件进行写入。

# 提取出50组用于量化的数据
for j in range(0, num_files):
    np.save(file="./data/quant/pdc%d.npy"%j, arr=X_test[j, :])  #将X_test的第j行保存在test.npy中
    file.write("./data/quant/pdc%d.npy\n"%j)    #使得dataset.txt中写每个量化的npy的文件路径

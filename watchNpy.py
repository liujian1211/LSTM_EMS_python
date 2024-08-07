import numpy as np

# 加载npy文件

for i in range(0,50):
    # index = i
    data = np.load(f'/home/ai/liujian/LSTM_RK3568/python_code/data/quant/qac'+ str(i) +'.npy')

# 打印npy文件中的内容
    print(data)
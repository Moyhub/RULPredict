import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

filename = 'data/train_FD001.txt'
sensor_data = np.loadtxt(filename)
print(sensor_data.shape)
#去除不相干维度
sensor_data = np.delete(sensor_data,[1,2,3,4,5,9,10,14,20,22,23],axis=1)
print(sensor_data.shape)
sensor_data = sensor_data.transpose(1,0)
#归一化
for i in range(1,sensor_data.shape[0]):
    sensor_data[i] = (sensor_data[i]-min(sensor_data[i]))/(max(sensor_data[i]) - min(sensor_data[i]))

choose = np.random.randint(1,sensor_data[0,-1],1)
choose =29
index = sensor_data[0].tolist().index(choose)
index_end = sensor_data[0].tolist().index(choose+1)
sensor_data = sensor_data + np.random.uniform(-0.1,0.1,sensor_data.shape)
print(choose,index,index_end)
plt.figure()
for i in range(1,15):
    plt.title("unit{},第{}个维度,加入噪声后".format(choose,i))
    plt.plot(range(index_end-index),sensor_data[i,index:index_end])
    plt.show()
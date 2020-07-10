import random
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

filename = 'data/train_FD001.txt'
#filename = 'data/test_FD001.txt'
index_line = []
random_num = []
index_num= []
#获取包含所有行的数据
with open(filename,'r') as file_to_read:
    lines = file_to_read.readlines()
print(len(lines))

#获取unit编号list
for line in lines:
    curline = line.strip().split(" ")
    floatLine =list(map(float,curline))#将数据直接转化成float类型
    index_line.append(floatLine[0])
print(len(index_line))

random_num = [60]
for i in random_num:
    temp = []
    if i != 100:
        temp.append(index_line.index(i))
        temp.append(index_line.index(i+1)-1)
    index_num.append(temp)
print(index_num)

#获取该随机unit 21个参数的曲线
np.random.seed(sum(map(ord,"aesthetics")))
for i in range(21):
    sensor_data = []
    time = []
    for line in lines:
        curline = line.strip().split(" ")
        floatLine = list(map(float,curline))
        sensor_data.append( round(floatLine[i+5],2) )
        time.append(round (floatLine[1]) )
    sensor_data = np.array(sensor_data)
    lin_data = (sensor_data-min(sensor_data))/(max(sensor_data)-min(sensor_data))
    plt.title("第{}个传感器数据变化趋势，当前unit为{}".format(i+1,random_num[0]))
    #plt.subplot(2, 3, 1)
    #plt.plot(time[index_num[0][0]:index_num[0][1]+1], sensor_data[index_num[0][0]:index_num[0][1]+1])
    plt.plot(time[index_num[0][0]:index_num[0][1]+1], lin_data[index_num[0][0]:index_num[0][1]+1])
    plt.show()






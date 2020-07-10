import numpy as np
import random
import copy
import math
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

class Dataprocess():
    def __init__(self,RUL_max ,BATCH_SIZE,data_serial):
        self.train_name = "data/train_FD00"+str(data_serial)+".txt"
        self.test_name = "data/test_FD00"+str(data_serial)+".txt"
        self.RUL_name = "data/RUL_FD00"+str(data_serial)+".txt"
        print("我们使用的数据集是:",self.train_name)
        self.BATCH_SIZE = BATCH_SIZE
        print("此时的Batch_size是",BATCH_SIZE)
        self.RUL_max = RUL_max
    def GetTrainBatchData(self):
        print("最大剩余寿命:",self.RUL_max)
        sensor_data_train,sensor_data_test,index_line_train,index_line_test = self.TurnToList()
        test_data_rul = self.GetRul()
        train_data,train_data_rul,train_seq_len = self.GetTrainLabel(sensor_data_train,index_line_train,self.RUL_max)
        test_data,test_data_rul,test_seq_len = self.GetTestData(sensor_data_test,index_line_test,test_data_rul)
        print("训练集长度",len(train_data))
        Max_Min=[1,0]
        train_data_rul,Max_Min = self.Normalization(train_data_rul,test_data_rul)
        batch_train_data, batch_train_data_rul, batch_train_seq_len = self.GetBatch(train_data,train_data_rul,train_seq_len,self.BATCH_SIZE)
        batch_test_data,batch_test_data_rul,batch_test_seq_len = self.GetBatch(test_data,test_data_rul,test_seq_len,self.BATCH_SIZE)
        batch_train_mask = self.GetMask(batch_train_data,batch_train_seq_len)
        batch_test_mask = self.GetMask(batch_test_data,batch_test_seq_len)
        return batch_train_data,batch_train_data_rul,batch_train_mask,batch_test_data,batch_test_data_rul,batch_test_mask,Max_Min
    def TurnToList(self):
        index_line_train = []
        index_line_test = []
        sensor_data_train = [[] for i in range(14)]
        sensor_data_test = [[] for j in range(14)]
        with open(self.train_name, 'r') as file_to_read:
            lines = file_to_read.readlines()
        for line in lines:
            curline = line.strip().split(" ")
            floatLine = list(map(float, curline))
            index_line_train.append(floatLine[0])
            j = 0
            for i in range(5, 26):
                if i in [5, 9, 10, 14, 20, 22, 23]:
                    continue
                else:
                    sensor_data_train[j].append(floatLine[i])
                    j += 1
        with open(self.test_name, 'r') as file_to_read:
            lines = file_to_read.readlines()
        for line in lines:
            curline = line.strip().split(" ")
            floatLine = list(map(float, curline))
            index_line_test.append(floatLine[0])
            j = 0
            for i in range(5, 26):
                if i in [5, 9, 10, 14, 20, 22, 23]:
                    continue
                else:
                    sensor_data_test[j].append(floatLine[i])
                    j += 1
        sensor_data_train = np.array(sensor_data_train)
        sensor_data_test = np.array(sensor_data_test)
        #对每一个维度进行归一化
        for i in range(len(sensor_data_train)):
            max_sensor = max(max(sensor_data_train[i]),max(sensor_data_test[i]))
            min_sensor = min(min(sensor_data_train[i]),min(sensor_data_test[i]))
            sensor_data_train[i] = (sensor_data_train[i]-min_sensor)/(max_sensor-min_sensor)
            sensor_data_test[i] = (sensor_data_test[i]-min_sensor) / (max_sensor-min_sensor)
        return sensor_data_train,sensor_data_test,index_line_train,index_line_test
    def GetRul(self):
        train_data_rul = []
        with open(self.RUL_name,'r') as file_to_read:
            lines = file_to_read.readlines()
        for line in lines:
            curline = line.strip().split(" ")
            floatLine = list(map(float,curline))
            train_data_rul.append(floatLine[0])
        train_data_rul = np.array(train_data_rul)
        return train_data_rul
    def GetTrainLabel(self,sensor_data,index_line,Rul_max):
        train_data = []
        seq_len = []
        train_data_rul = []
        print("训练集长度:",int(index_line[len(index_line) - 1]))
        for i in range(1, int(index_line[len(index_line) - 1]) + 1):
            if i == int(index_line[len(index_line) - 1]):
                max_seq = len(index_line) - 1
                min_seq = index_line.index(i)
            else:
                max_seq = index_line.index(i + 1) - 1
                min_seq = index_line.index(i)
            start = min_seq + math.floor((max_seq - min_seq) / 12)#8
            end = max_seq
            list_breakpoint = random.sample(range(start, end),end-start )#math.ceil((max_seq-min_seq)/2)
            for mybreakpoint in list_breakpoint:
                temp = sensor_data[:, min_seq:mybreakpoint].tolist()  # 不包含mybreakpoint，包含min_seq,这是
                temp_rul = max_seq - mybreakpoint + 1
                if temp_rul >=Rul_max:
                    temp_rul = Rul_max
                train_data.append(temp)
                seq_len.append(len(temp[1]))
                train_data_rul.append(temp_rul)
        index = np.argsort(np.array(seq_len))
        train_data = np.array(train_data)
        train_data = (train_data)[index]  # 这里还没有分块，5000*元素，每一个元素都是一个14*seq,由于seq不相同，所以现在的train_data.shape(5000.14)
        train_data_rul = np.array(train_data_rul)[index]  # 同每一个元素就是一个数字，5000个
        seq_len.sort()
        return train_data,train_data_rul,seq_len
    def GetTestData(self,sensor_data,index_line,test_data_rul):
        seq_len = []
        test_data = []
        print("测试集长度:",(index_line[len(index_line) - 1]))
        for i in range(1, int(index_line[len(index_line) - 1]) + 1):
            if i == int(index_line[len(index_line) - 1]):
                max_seq = len(index_line) - 1
                min_seq = index_line.index(i)
            else:
                max_seq = index_line.index(i + 1) - 1
                min_seq = index_line.index(i)
            temp = sensor_data[:, min_seq:max_seq + 1].tolist()
            test_data.append(temp)
            seq_len.append(len(temp[1]))
        test_data = np.array(test_data)
        index = np.argsort(np.array(seq_len))
        test_data = test_data[index]
        test_data_rul = test_data_rul[index]
        seq_len.sort()
        return test_data,test_data_rul,seq_len
    def GetBatch(self,train_data,train_data_rul,seq_len,BATCH_SIZE):
        # 根据BATCH_SIZE进行分块
        num_batches = np.ceil(len(train_data) / BATCH_SIZE).astype('int')
        batch_train_data = []  # 每一个元素都是一个三维矩阵
        batch_train_data_rul = []  # 每一个元素都是一个32*1的剩余寿命
        batch_seq_len = []  # 每一个元素都是一个32*1的长度
        for i in range(num_batches):
            pad_seq_max = max(seq_len[i * BATCH_SIZE: min(len(train_data), (i + 1) * BATCH_SIZE)])  # 找出该batch中seq最长长度
            batch_temp = train_data[i * BATCH_SIZE: min(len(train_data), (i + 1) * BATCH_SIZE)]     # 32*14
            batch3D = np.empty(shape=(len(batch_temp), len(batch_temp[0]), pad_seq_max),dtype=float)             # 创建一个batch_size * feature * max_seq
            for j in range(len(batch_temp)):            # 为每一个batch进行填充len(batch_temp) = 32)
                data = np.array(batch_temp[j].tolist(), dtype=float)  # data numpy 14*63
                padding_matrix = np.zeros((len(data), pad_seq_max - len(data[0])))
                data = np.append(data, padding_matrix, axis=1)  # data numpy 14*pad_seq_max
                batch3D[j] = data                            # batch[j].shape = (14,pad_seq_max),batch.shape = (32,14,pad_seq_max)
            batch_train_data.append(batch3D)                 # batch3D.shape=(8,14,68)
            batch_train_data_rul.append(train_data_rul[i * BATCH_SIZE: min(len(train_data), (i + 1) * BATCH_SIZE)])
            batch_seq_len.append(seq_len[i * BATCH_SIZE: min(len(train_data), (i + 1) * BATCH_SIZE)])
        print("训练/测试集batch个数:",len(batch_train_data_rul), len(batch_train_data), len(batch_seq_len))
        return batch_train_data,batch_train_data_rul,batch_seq_len
    def GetMask(self,batch_train_data,batch_seq_len):
        batch_mask = []
        for i in range(len(batch_train_data)):
            max_seq_pad = batch_train_data[i].shape[2]
            mask = torch.zeros([batch_train_data[i].shape[0], max_seq_pad])
            for j in range(batch_train_data[i].shape[0]):
                length_seq = batch_seq_len[i][j]
                mask[j][length_seq - 1] = 1
                # for k in range(2,6):
                #     mask[j][length_seq-k]=1
            batch_mask.append(mask)
        return batch_mask
    def Normalization(self,train_data_rul,test_data_rul):
        Max_Min = []
        Max_Min.append( max(max(test_data_rul),max(train_data_rul)) )
        Max_Min.append( min(min(test_data_rul),min(train_data_rul)) )
        train = ( train_data_rul-Max_Min[1] )/( Max_Min[0]-Max_Min[1] )
        print(Max_Min)
        return train,Max_Min
import numpy as np
import random
import copy
import math
import torch
import matplotlib.pyplot as plt

def DataProcess(filename ='data/test_FD001.txt',RUL_name=None,max_min = None,BATCH_SIZE =32,Test = False): #Test=False时,max_min必须为None
    index_line = []                          #unit
    sensor_data = [[] for i in range(14)]    #sensor数据
    if max_min == None:
        max_min = copy.deepcopy(sensor_data)    #这才是深复制，直接赋值和.copy均为浅复制，max_min记录最大值与最小值
    #得到的训练数据和剩余寿命list
    train_data = []
    seq_len = [] #same size with train_data
    train_data_rul= []
    #获取每一行的数据，并将其转化为Float类型，其中第1,5,6,10,16,18,19号传感器数据舍弃
    with open(filename,'r') as file_to_read:
        lines = file_to_read.readlines()
    for line in lines:
        curline = line.strip().split(" ")
        floatLine =list(map(float,curline))
        index_line.append(floatLine[0])
        j=0
        for i in range(5,26):
            if i in [5,9,10,14,20,22,23]:
                continue
            else:
                sensor_data[j].append(floatLine[i])
                j+=1
    if RUL_name is not None:#for test data
        with open(RUL_name,'r') as file_to_read:
            lines = file_to_read.readlines()
        for line in lines:
            curline = line.strip().split(" ")
            floatLine = list(map(int,curline))
            train_data_rul.append(floatLine[0])

    '''到此我们已经将数据放进一个二维数组，并排除了不相关的维度下面进行加噪声的操作'''
    if Test == False:
        sensor_data = np.array(sensor_data)
        #对每一个维度进行归一化
        for i in range(len(sensor_data)):
            max_sensor = max(sensor_data[i])
            min_sensor = min(sensor_data[i])
            max_min[i].append(max_sensor)
            max_min[i].append(min_sensor)
            sensor_data[i] = (sensor_data[i]-min_sensor)/(max_sensor-min_sensor)
    else:
         sensor_data = np.array(sensor_data)
         for i in range(len(sensor_data)):
            sensor_data[i] = (sensor_data[i] - max_min[i][1]) / (max_min[i][0] - max_min[i][1])  #sensor_data np 14*20631
    #随机切割数据并构造标签
    for i in range(1,int(index_line[len(index_line)-1])+1): #i从1-100,我将在每一个unit中随机切割出200个emample
        new_example = []
        if i == 100:
            max_seq = len(index_line)-1
            min_seq = index_line.index(i)
        else:
            max_seq = index_line.index(i+1)-1
            min_seq = index_line.index(i)
        if Test == False:
            start = min_seq + math.floor((max_seq - min_seq) / 6)
            end = max_seq-math.floor((max_seq-min_seq)/6)
            list_breakpoint = random.sample(range(start,end),end-start)
            for mybreakpoint in list_breakpoint:
                temp = sensor_data[:,min_seq:mybreakpoint].tolist() #不包含mybreakpoint，包含min_seq,这是
                temp_rul = max_seq - mybreakpoint + 1
                train_data.append(temp)
                seq_len.append( len(temp[1]) )
                train_data_rul.append(temp_rul)  #train_data_rul 为list,20000
        else:#for test data
            temp = sensor_data[:,min_seq:max_seq+1].tolist()
            train_data.append(temp)              #train_data_rul 为list,100
            seq_len.append(len(temp[1]))

    #切割标签后现在有5000或者test的100条数据，下面是根据seq长度对其进行排序
    index = np.argsort( np.array(seq_len) )
    train_data = np.array(train_data)
    train_data = (train_data)[index]                #这里还没有分块，5000*元素，每一个元素都是一个14*seq,由于seq不相同，所以现在的train_data.shape(5000.14)
    train_data_rul = np.array(train_data_rul)[index]#同每一个元素就是一个数字，5000个
    seq_len.sort()
    print(train_data.shape)
    #根据BATCH_SIZE进行分块
    num_batches = np.ceil(len(train_data)/BATCH_SIZE).astype('int')
    batch_train_data = []                  #每一个元素都是一个三维矩阵
    batch_train_data_rul = []              #每一个元素都是一个32*1的剩余寿命
    batch_seq_len = []                     #每一个元素都是一个32*1的长度
    for i in range(num_batches):
        pad_seq_max = max( seq_len[i*BATCH_SIZE : min(len(train_data),(i+1)*BATCH_SIZE)] )           #找出该batch中seq最长长度
        batch_temp = train_data[i*BATCH_SIZE : min(len(train_data),(i+1)*BATCH_SIZE)]                #32*14
        batch3D = np.empty(shape=(len(batch_temp),len(batch_temp[0]),pad_seq_max),dtype=float)       #创建一个batch_size * feature * max_seq
        for j in range(len(batch_temp)):                                                             #为每一个batch进行填充len(batch_temp) = 32)
            data = np.array(batch_temp[j].tolist(),dtype=float)     #data numpy 14*63
            padding_matrix = np.zeros((len(data),pad_seq_max-len(data[0])))
            data = np.append(data,padding_matrix,axis=1)            #data numpy 14*pad_seq_max
            batch3D[j] = data                                       #batch[j].shape = (14,pad_seq_max),batch.shape = (32,14,pad_seq_max)
        batch_train_data.append(batch3D)                            #batch3D.shape=(8,14,68)
        batch_train_data_rul.append(train_data_rul[i*BATCH_SIZE : min(len(train_data),(i+1)*BATCH_SIZE)])
        batch_seq_len.append(seq_len[i*BATCH_SIZE : min(len(train_data),(i+1)*BATCH_SIZE)])
    print(len(batch_train_data_rul), len(batch_train_data),len(batch_seq_len))
    #构建batch_mask---只使用T时刻
    batch_mask = []
    for i in range(num_batches):
        max_seq_pad = batch_train_data[i].shape[2]
        mask = torch.zeros([batch_train_data[i].shape[0],max_seq_pad])
        for j in range(batch_train_data[i].shape[0]):
            length_seq = batch_seq_len[i][j]
            mask[j][length_seq-1] = 1
        batch_mask.append(mask)
    #构建batch_mask---使用全部时刻
    # batch_mask = []
    # for i in range(num_batches):
    #     max_seq_pad = batch_train_data[i].shape[2]
    #     mask = torch.zeros([batch_train_data[i].shape[0],max_seq_pad])
    #     for j in range(batch_train_data[i].shape[0]):
    #         length_seq = batch_seq_len[i][j]
    #         mask[j][:length_seq] = 1
    #     batch_mask.append(mask)
    #构建batch_mask---使用随机时刻
    # batch_mask = []
    # for i in range(num_batches):
    #     max_seq_pad = batch_train_data[i].shape[2]
    #     mask = torch.zeros([batch_train_data[i].shape[0],max_seq_pad])
    #     for j in range(batch_train_data[i].shape[0]):
    #         length_seq = batch_seq_len[i][j]
    #         mask[j][length_seq - 1] = 1
    #         randomnum = random.sample(range(0,length_seq),10)
    #         for k in randomnum:
    #             mask[k] = 1
    #     batch_mask.append(mask)
    #改良batch_train_data_rul
    # batch_rul = []
    # for i in range(num_batches):
    #     max_seq_pad = batch_train_data[i].shape[2]
    #     rul = torch.zeros([batch_train_data[i].shape[0],max_seq_pad])
    #     for j in range(batch_train_data[i].shape[0]):
    #         Temp = torch.arange(batch_train_data_rul[i][j]+batch_seq_len[i][j]-1,batch_train_data_rul[i][j]-1,-1)
    #         rul[j][:batch_seq_len[i][j]] = Temp
    #     batch_rul.append(rul)
    # batch_train_data_rul = batch_rul
    return batch_train_data,batch_train_data_rul,batch_mask,max_min

class ProcessTestData():
    def __init__(self):
        self.test_name = "data/test_FD001.txt"
        self.RUL_name = "data/RUL_FD001.txt"
    def GetTrainBatchData(self):
        sensor_data,index_line = self.TurnTolist()
        train_data_rul = self.GetRul()
        train_data,seq_len = self.GetPart(sensor_data,index_line)
        batch_train_data,batch_train_data_rul,batch_seq_len = self.GetBatch(train_data,train_data_rul,seq_len)
        batch_mask = self.GetMask(batch_train_data,batch_seq_len)
        return batch_train_data,batch_train_data_rul,batch_mask
    def TurnTolist(self):
        index_line = []  # unit
        sensor_data = [[] for i in range(14)]  # sensor数据
        max_min = []
        with open(self.test_name, 'r') as file_to_read:
            lines = file_to_read.readlines()
        for line in lines:
            curline = line.strip().split(" ")
            floatLine = list(map(float, curline))
            index_line.append(floatLine[0])
            j = 0
            for i in range(5, 26):
                if i in [5, 9, 10, 14, 20, 22, 23]:
                    continue
                else:
                    sensor_data[j].append(floatLine[i])
                    j += 1
        sensor_data = np.array(sensor_data)
        #对每一个维度进行归一化
        for i in range(len(sensor_data)):
            max_sensor = max(sensor_data[i])
            min_sensor = min(sensor_data[i])
            sensor_data[i] = (sensor_data[i]-min_sensor)/(max_sensor-min_sensor)
        return sensor_data,index_line
    def GetRul(self):
        train_data_rul = []
        with open(self.RUL_name,'r') as file_to_read:
            lines = file_to_read.readlines()
        for line in lines:
            curline = line.strip().split(" ")
            floatLine = list(map(int,curline))
            train_data_rul.append(floatLine[0])
        return train_data_rul
    def GetPart(self,sensor_data,index_line):
        seq_len = []
        train_data = []
        for i in range(1, 101):
            if i == 100:
                max_seq = len(index_line) - 1
                min_seq = index_line.index(i)
            else:
                max_seq = index_line.index(i + 1) - 1
                min_seq = index_line.index(i)
            temp = sensor_data[:, min_seq:max_seq + 1].tolist()
            train_data.append(temp)  # train_data_rul 为list,100
            seq_len.append(len(temp[1]))
        return train_data,seq_len
    def GetBatch(self,train_data,train_data_rul,seq_len):
        BATCH_SIZE=4
        train_data = np.array(train_data)
        train_data_rul = np.array(train_data_rul)
        # 根据BATCH_SIZE进行分块
        num_batches = np.ceil(len(train_data) / BATCH_SIZE).astype('int')
        batch_train_data = []  # 每一个元素都是一个三维矩阵
        batch_train_data_rul = []  # 每一个元素都是一个32*1的剩余寿命
        batch_seq_len = []  # 每一个元素都是一个32*1的长度
        for i in range(num_batches):
            pad_seq_max = max(seq_len[i * BATCH_SIZE: min(len(train_data), (i + 1) * BATCH_SIZE)])  # 找出该batch中seq最长长度
            batch_temp = train_data[i * BATCH_SIZE: min(len(train_data), (i + 1) * BATCH_SIZE)]  # 32*14
            batch3D = np.empty(shape=(len(batch_temp), len(batch_temp[0]), pad_seq_max),
                               dtype=float)  # 创建一个batch_size * feature * max_seq
            for j in range(len(batch_temp)):  # 为每一个batch进行填充len(batch_temp) = 32)
                data = np.array(batch_temp[j].tolist(), dtype=float)  # data numpy 14*63
                padding_matrix = np.zeros((len(data), pad_seq_max - len(data[0])))
                data = np.append(data, padding_matrix, axis=1)  # data numpy 14*pad_seq_max
                batch3D[j] = data  # batch[j].shape = (14,pad_seq_max),batch.shape = (32,14,pad_seq_max)
            batch_train_data.append(batch3D)  # batch3D.shape=(8,14,68)
            batch_train_data_rul.append(train_data_rul[i * BATCH_SIZE: min(len(train_data), (i + 1) * BATCH_SIZE)])
            batch_seq_len.append(seq_len[i * BATCH_SIZE: min(len(train_data), (i + 1) * BATCH_SIZE)])
        print(len(batch_train_data_rul), len(batch_train_data), len(batch_seq_len))
        return batch_train_data,batch_train_data_rul,batch_seq_len
    def GetMask(self,batch_train_data,batch_seq_len):
        batch_mask = []
        for i in range(len(batch_train_data)):
            max_seq_pad = batch_train_data[i].shape[2]
            mask = torch.zeros([batch_train_data[i].shape[0], max_seq_pad])
            for j in range(batch_train_data[i].shape[0]):
                length_seq = batch_seq_len[i][j]
                mask[j][length_seq - 1] = 1
            batch_mask.append(mask)
        return  batch_mask
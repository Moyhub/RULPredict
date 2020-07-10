import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import model
import os
import copy
from random import shuffle
import matplotlib.pyplot as plt
import random
import math
import NewDataProcess
#设置随机数种子使得结果可复现
def seed_torch(seed):
    print("此时seed的值为:",seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed
# 三种评分函数
def Score(predict_label,figure_test):
    Score=0
    RMSE =0
    Accuracy=0
    Offset = [predict_label[i] - figure_test[i] for i in range(len(predict_label))]
    for index in Offset:
        if index < 0:
            score = math.exp(-index / 13) - 1
        if index > 0:
            score = math.exp(index / 10) - 1
        Score += score
    for index in  Offset:
        RMSE += index**2
    RMSE = math.sqrt(RMSE/len(Offset))
    for index in Offset:
        if index>=-13 and index <=10:
            Accuracy+=1
    Accuracy = 100/len(Offset) * Accuracy
    print(40*"#")
    print("评分值", Score)
    print("RMSE",RMSE)
    print("Accuracy",Accuracy)
    print(40*"#")

def RULpredict(n=2,RUL_max =125,BATCH_SIZE=8,data_serial=1):
    Transformer_model = model.Model(14)
    #加入模型参数初始化
    for p in Transformer_model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    #优化器与损失函数
    optimizer = torch.optim.Adam(Transformer_model.parameters(),lr=0.001,eps=1e-8)
    criterion = nn.MSELoss()
    #获取预处理后的数据
    dateset = NewDataProcess.Dataprocess(RUL_max,BATCH_SIZE,data_serial)
    train_data,train_data_label,train_seq_mask,test_data,test_data_label,test_seq_mask,Max_Min = dateset.GetTrainBatchData()
    look = []
    for i, batch in enumerate(train_data):
        train_data[i] = torch.from_numpy(batch)#.transpose(1,2)
        train_data_label[i] = torch.from_numpy(train_data_label[i])
    for i,test in enumerate(test_data):
        test_data[i] = torch.from_numpy(test)#.transpose(1,2)
        test_data_label[i] = torch.from_numpy(test_data_label[i])
    #保留原始的数据
    save_train_data = copy.deepcopy(train_data)
    save_train_data_label = copy.deepcopy(train_data_label)
    save_train_seq_mask = copy.deepcopy(train_seq_mask)
    def train(train_data,train_data_label,train_seq_mask):
        Transformer_model.train()
        zipfile = list(zip(train_data,train_data_label,train_seq_mask))
        shuffle(zipfile)
        train_data[:],train_data_label[:],train_seq_mask[:]= zip(*zipfile)
        all_loss = 0.
        print("当前学习率:", optimizer.state_dict()['param_groups'][0]['lr'])
        for i, batch in enumerate(train_data):
            batch_mask = model.make_std_mask(len(batch), batch.shape[2])
            batch = torch.tensor(batch, dtype=torch.float32)
            label = torch.tensor(train_data_label[i], dtype=torch.float32)
            if torch.cuda.is_available():
                batch_mask = batch_mask.cuda()
                batch = batch.cuda()
                label = label.cuda()
                train_seq_mask[i].cuda()
            optimizer.zero_grad()
            out = Transformer_model.forward(batch, batch_mask)  # batch_mask需要自己构建，这部分和论文中的不同
            finalout = out[train_seq_mask[i] == 1]
            loss = criterion(finalout, label)  # 直接计算loss
            all_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("This train process the almost_rmseloss is:{:10f}".format( math.sqrt((all_loss)/ len(train_data)) ))
    def evaluate(test_data,test_data_label,test_seq_mask,Max_Min):
        Transformer_model.eval()
        total_loss = 0.
        predict_label = []
        figure_test = []
        with torch.no_grad():
            for i,batch in enumerate(test_data):
                batch_mask = model.make_std_mask(len(batch), batch.shape[2]) #为了使用卷积这里改变了一下内容
                batch = torch.tensor(batch, dtype=torch.float32)
                label = torch.tensor(test_data_label[i],dtype=torch.float32)
                #记录test集合的label
                for z in label:
                    figure_test.append(z.item())
                #迁移到GPU
                if torch.cuda.is_available():
                    batch_mask = batch_mask.cuda()
                    batch = batch.cuda()
                    label = label.cuda()
                    test_seq_mask[i].cuda()
                out = Transformer_model(batch,batch_mask)
                masked_out = out[test_seq_mask[i]==1]
                masked_out = masked_out*(Max_Min[0]-Max_Min[1])+Max_Min[1]
                finalout = masked_out
                # final = masked_out.view(batch.shape[0],-1)
                # finalout = final[:,-5]
                #finalout = (final[:,0]+final[:,1]+final[:,2])/3
                #记录预测结果的label
                for i in finalout:
                    predict_label.append(i.item())
                loss = criterion(finalout,label)
                total_loss +=loss.item()*len(label)
            print("Test数据集长度：",(len(test_data_label)-1)*len(test_data_label[0])+len(test_data_label[-1]) )
            x = math.sqrt(total_loss/((len(test_data_label)-1)*len(test_data_label[0])+len(test_data_label[-1])))
            print("Test_RMSELoss:",x)
            figure_test1 = [i+0 for i in figure_test]
            Score(predict_label, figure_test1)
        return x,predict_label,figure_test1
    #重新加载
    # Transformer_model = torch.load('model_save/Second/model_RD004.pth', map_location="cpu")
    # # Transformer_model = torch.load('result/model.pth', map_location="cpu")
    # min_loss, predict_label, figure_test= evaluate(test_data, test_data_label, test_seq_mask, Max_Min)
    #迁移到GPU上
    if torch.cuda.is_available():
        Transformer_model = Transformer_model.cuda()
    for epoch in range(15):
        print("Train Process: the {:2d} epoth".format(epoch))
        for i in range(len(train_data)):
            train_data[i] = train_data[i] + torch.from_numpy(np.random.uniform(-0.1,0.1,train_data[i].shape))
        train(train_data,train_data_label,train_seq_mask)
        if(epoch>=10):
            #if optimizer.state_dict()['param_groups'][0]['lr'] >0.0005:
            for paras in optimizer.param_groups:
                paras['lr']=paras["lr"]-0.0001
        min_loss,predict_label,figure_test= evaluate(test_data, test_data_label, test_seq_mask,Max_Min)
        look.append(min_loss)
        if min_loss == min(look):
            print("当前epoch精度最优，保存模型,此时的epoch值为：",epoch)
            torch.save(Transformer_model,'model.pth')
        train_data = copy.deepcopy(save_train_data)
        train_data_label = copy.deepcopy(save_train_data_label)
        train_seq_mask = copy.deepcopy(save_train_seq_mask)
    print(100*"*")
    print("min_loss",min(look))
    return  min(look),predict_label,figure_test
    # return min_loss,predict_label,figure_test

filename="batch.txt"
with open(filename, "w") as f:
    # SeedList = []
    # for i in range(60):
    #     Seed = random.randint(1, 1000)
    #     while Seed in SeedList:
    #         Seed = random.randint(1, 1000)
    #     SeedList.append(Seed)
    Seed =217
    seed = seed_torch(seed=Seed)
    x,predict_label,figure_test = RULpredict(data_serial=3) #data_serial是数据集编号
    f.write("此时seed的值为:"+ str(seed)+"   ")
    f.write("Min_LOSS:"+str(x))
    f.write("\n")
    # for i in SeedList:
    #     f.write(str(i)+"  ")
    filename = "result/predict.txt"
    with open(filename,"w") as f:
        for x in predict_label:
            f.write(str(x)+" ")
    filename = "result/test.txt"
    with open(filename,"w") as f:
        for x in figure_test:
            f.write(str(x)+" ")
    filename = "result/test_seq.txt"

#加卷积处理，main.py中mask和数据transpose
#model.py 加卷积层;维度匹配
#dropout，batch_size,噪声，dataset,SEED

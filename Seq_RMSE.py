import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

filename = "result/test.txt"
test_label = np.loadtxt(filename)
predict_label = np.loadtxt("result/predict.txt")
seq = np.loadtxt("result/test_seq.txt").tolist()

Rmse = []
seq_index = []
score = 0

def RMSE(x,y):
    return math.sqrt((x-y)**2)
seq_norepeat = list(set(seq))
seq_norepeat.sort()
for i in seq_norepeat:
    seq_index.append(seq.index(i))

print("无重复seq个数",len(seq_norepeat))
for i in range(len(seq_norepeat)):
    if i != len(seq_norepeat)-1:
        Min = seq.index(seq_norepeat[i])
        Max = seq.index(seq_norepeat[i+1])
    else:
        Min = seq.index(seq_norepeat[i])
        Max = len(seq)
    for k in range(Min,Max):
        score+=RMSE(predict_label[k],test_label[k])
    score= score/(Max-Min)
    Rmse.append(score)
    score = 0
plt.figure()
plt.plot(seq_norepeat,Rmse)
plt.show()





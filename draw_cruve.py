import matplotlib.pyplot as plt
import numpy as np
import math


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
    return Score,RMSE,Accuracy

filename = "result/test.txt"
test_label = np.loadtxt(filename)
predict_label = np.loadtxt("result/predict.txt")
S = []
R = []
A = []
for i in range(11):
    print("将RUL_est减低")
    predict_label = predict_label - i/10
    score,rmse,acc=Score(predict_label,test_label)
    S.append(score)
    R.append(rmse)
    A.append(acc)
x = range(11)
plt.subplot(1,3,1)
plt.plot(x,S)
plt.subplot(1,3,2)
plt.plot(x,R)
plt.subplot(1,3,3)
plt.plot(x,A)
plt.show()

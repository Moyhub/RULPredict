import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

filename = "result/test.txt"
test_label = np.loadtxt(filename)
predict_label = np.loadtxt("result/predict.txt")

index = np.argsort(-test_label)
test_label = test_label[index]
predict_label = predict_label[index]
piece_wise = []
#分段线性
for i in test_label:
    if i>125:
        piece_wise.append(125)
    else:
        piece_wise.append(i)
#曲线拟合
x = range(len(test_label))
func = np.polyfit(x,predict_label,4)
fun = np.poly1d(func)
yvals1 = fun(x)

plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.grid(linestyle="-.")
plt.xlabel("Unit")
plt.xlim([0,len(test_label//10*10+10)])
plt.ylabel("Remaining Useful Life(Cycle)")
plt.ylim([0,150])
# plt.plot(x,test_label,"b")
plt.plot(x,predict_label,"y")
plt.plot(x,piece_wise,"r")
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
         }
plt.legend(["Predicted RUL","Ground-True RUL (Piece-Wise)"],loc="lower left",prop=font1)
# plt.scatter(x,predict_label,marker="x",c="r")
plt.show()
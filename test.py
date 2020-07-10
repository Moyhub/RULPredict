import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import math
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

# RMSE 函数图像
h= np.arange(-50,50,0.1)
y1=[]
for i in h:
    y1.append(math.fabs(i))
y2=[]
for i in h:
    if i<0:
        score = math.exp(-i/13)-1
        y2.append(score)
    if i>0:
        score = math.exp(i/10)-1
        y2.append(score)

plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }
plt.xlabel("h",font2)
plt.ylabel("Value of metrics(RMSE/Scoring Function)",font2)
plt.xlim(-50,50)
plt.ylim(0,160)
plt.grid(linestyle="-.")
plt.plot(h,y1,"b")
plt.plot(h,y2,"r-")
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }
plt.legend(["RMSE","Scoring Function"],prop=font1)
plt.show()

# x = np.arange(200,0,-1)
# x_1 = range(200)
# y= []
# for i in x:
#     if i >125:
#         y.append(125)
#     else:
#         y.append(i)
# plt.figure()
# plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
# plt.xlim([0,200])
# plt.ylim([0,200])
# plt.grid(linestyle="-.")
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 14,
#          }
# plt.ylabel("Remaining Useful Life(RUL)",font2)
# plt.xlabel("Time Cycle",font2)
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 14,
#          }
# plt.plot(x_1,x,"b")
# plt.plot(x_1,y,"r",linestyle="--")
# plt.legend(["Actual RUL","Piece-Wise RUL"],prop=font1)
# plt.show()
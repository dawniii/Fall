"""
model:HMM
data:2022.11
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
f='E:/Fall/UPFall/data/CompleteDataSet.csv'
df=pd.read_csv(f,header=None,skiprows=2)
df=df[[1,2,3,43,44,45,46]]
columns=['Ankle_ACC_X','Ankle_ACC_Y','Ankle_ACC_Z','Subject','Activity','Trial','Tag']
df = pd.DataFrame(df.values, columns=columns)
TIME_STEPS=20 #根据csv文件,1s20个数据
FEATURES=30
step=15
segments=[]
labels=[]
for subject in range(1):
    for activity in range(11):
        for trial in range(3):
            dk = pd.DataFrame()
            dk=df.loc[df['Activity']==activity+1]
            dk=dk.loc[df['Subject']==subject+1]
            dk=dk.loc[df['Trial']==trial+1]
            dk=dk.loc[df['Tag']==activity+1]
            #调整dk的标号
            dk.reset_index(drop=True, inplace=True)
           # scale_columns = ['Ankle_ACC_X','Ankle_ACC_Y','Ankle_ACC_Z']
            # 归一化，映射到-1到1之间
           # scaler = MinMaxScaler(feature_range=(-1, 1))
           # scaler = scaler.fit(dk[scale_columns])
           # dk.loc[:, scale_columns] = scaler.transform(dk[scale_columns].to_numpy())
            for i in range(0, len(dk) - TIME_STEPS, step):
                x1 = dk['Ankle_ACC_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y1 = dk['Ankle_ACC_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z1 = dk['Ankle_ACC_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                segments.append(np.hstack((x1, y1, z1)))
                #segments.append(np.hstack((x1, y1, z1, x2, y2, z2)))
                if(activity<5):
                    labels.append(1)
                else:
                    labels.append(0)
print(segments)
print(labels)
adl_list=[]
fall_list=[]
for i in range(len(segments)):
    if(labels[i]==0):
        adl_list.append(segments[i])
    else:
        fall_list.append(segments[i])

# 求合成加速度，每个segments[i]有20个值
def all_acc(segments):
    hmm_list = []
    for i in range(len(segments)):#遍历每个列表
        list_per_win=[] #200
        for j in range(len(segments[i])):
            sum=0
            for k in range(3):
                sum += (segments[i][j][k]*segments[i][j][k])
            list_per_win.append(np.array(math.sqrt(sum)))
        hmm_list.append(list_per_win)
    return hmm_list
adl_hmm_list=all_acc(adl_list)
fall_hmm_list=all_acc(fall_list)
print('---')
# def process(per_list):
#     # 观测序列observ_list
#     observ_list=[]
#     for i in range(0,len(per_list),5):
#         observ_list.append(np.sum(per_list[i:i+5])/5) # 共40个
#     #特征化
#     f_list=[]
#     for j in range(len(observ_list)):
#         if(observ_list[j]<=1):
#             f_list.append([1])
#         elif(observ_list[j]>3):
#             f_list.append([3])
#         else:
#             f_list.append([2])
#     return f_list

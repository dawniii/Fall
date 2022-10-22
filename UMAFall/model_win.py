import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import stats
from tcn import TCN, tcn_full_summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import glob,os
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
from sklearn.model_selection import KFold

FEATURES =3
RANDOM_SEED = 42
TIME_STEPS=250
step=250


#ParentDir="E:/Fall_Detection/dataset/UMAFall_Dataset/"


#日常活动 0 跌倒 1
def Dataprocess(ParentDir,filepath):
    df = pd.read_csv(filepath, header=32, sep=';')
    df = df.loc[df[' Sensor Type'] == 0]# Accelerometer = 0
    df = df.loc[df[' Sensor ID'] == 0]# 4; ANKLE; SensorTag
    df.reset_index(drop=True, inplace=True)
    # if filepath[len(ParentDir) + 19:len(ParentDir) + 23] == 'Fall':
    #     df=df[300:800]
    # 归一化
    scale_columns = [' X-Axis',' Y-Axis',' Z-Axis']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(df[scale_columns])
    df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
    for i in range (0,len(df)-TIME_STEPS,step):
        x = df[' X-Axis'].values[i:i + TIME_STEPS].reshape(-1, 1)
        y = df[' Y-Axis'].values[i:i + TIME_STEPS].reshape(-1, 1)
        z = df[' Z-Axis'].values[i:i + TIME_STEPS].reshape(-1, 1)
        segment.append(np.hstack((x,y,z)))
    if filepath[len(ParentDir)+20:len(ParentDir)+23]=='ADL':
        label = len(segment)*[0]
        # 构建一个segments长的列表，值全为0
    if filepath[len(ParentDir)+20:len(ParentDir)+24]=='Fall':
        label = len(segment) * [1]
        #值全为1
    return segment,label
segments=[]
labels=[]
ParentDir="E:/Fall/UMAFall/data/UMAFall_Dataset"
file = glob.glob(os.path.join(ParentDir, "*.csv"))
for f in file:
    segment = []
    label = []
    Dataprocess(ParentDir,f)
    segments +=segment
    labels +=label
accx=[]
accy=[]
accz=[]
for i in range(len(segments)):
    accx.append(segments[i][0][0])
    accy.append(segments[i][0][1])
    accz.append(segments[i][0][2])
reshaped_segments =np.asarray(segments,dtype=np.float32).reshape(-1,TIME_STEPS,FEATURES)
labels=np.asarray(pd.get_dummies(labels),dtype=np.float32)#是利用pandas实现one hot encode的方式。
floder = KFold(n_splits=3, random_state=42, shuffle=True)
LOSS=[]
ACC=[]
TN=[]
FP=[]
FN=[]
TP=[]
Accuracy=[]
Precision=[]
Recall=[]
k_fold_num=0
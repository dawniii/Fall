import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import stats
from sklearn.model_selection import KFold
from tcn import TCN, tcn_full_summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import glob,os
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)
f='E:/Fall/UPFall/data/CompleteDataSet.csv'
df=pd.read_csv(f,header=None,skiprows=2)
df=df[[1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,25,26,27,29,30,31,32,33,34,43,44,45]]
columns=['Ankle_ACC_X','Ankle_ACC_Y','Ankle_ACC_Z','Ankle_ANG_X','Ankle_ANG_Y','Ankle_ANG_Z','RP_ACC_X','RP_ACC_Y','RP_ACC_Z','RP_ANG_X','RP_ANG_Y','RP_ANG_Z',
         'BELT_ACC_X', 'BELT_ACC_Y', 'BELT_ACC_Z', 'BELT_ANG_X', 'BELT_ANG_Y', 'BELT_ANG_Z','NECK_ACC_X','NECK_ACC_Y','NECK_ACC_Z','NECK_ANG_X','NECK_ANG_Y','NECK_ANG_Z','WRST_ACC_X','WRST_ACC_Y','WRST_ACC_Z','WRST_ANG_X','WRST_ANG_Y','WRST_ANG_Z',
         'Subject','Activity','Trial']
df = pd.DataFrame(df.values, columns=columns)
print()
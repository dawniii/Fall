import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import stats
from sklearn.model_selection import train_test_split
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import glob,os
from sklearn.model_selection import KFold
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
TIME_STEPS=100 #100HZ
FEATURES =12
RANDOM_SEED = 42
step=75

file='E:/Fall/Cogent/dataprocess/falls/subject_1'
df=pd.read_csv(file)
df.loc[df['annotation_1'].isin([2]),'annotation_1']=0
# df['annotation_1'].replace(2,0)#将所有的2元素替换为0，返回dataframe
# df = df[~df['annotation_1'].isin([2])]
#     # 调整df的标号
# df.reset_index(drop=True, inplace=True)
# i = 0
# start_list = []  # 统计所有跌倒的开始点
# end_list = []  # 统计所有跌倒的结束
# while (i < len(df)):
#     if (df['annotation_1'][i] == 1.0):  # 一次跌倒发生的开始
#         start_fall_index = i  # 一次跌倒发生的开始点
#         end = i + 1
#         while ((0.0 in list(df['annotation_1'][start_fall_index:end])) == False):
#             i = i + 1
#             end = i
#         end_fall_index = end - 2
#         start_list.append(start_fall_index)
#         end_list.append(end_fall_index)
#         # 找到峰值所在的坐标
#         max_id = df['ch_accel_x'][start_fall_index:end_fall_index + 1].idxmax()
#         print(start_fall_index, end_fall_index, df['ch_accel_x'][start_fall_index:end_fall_index + 1].max(),
#                 df['ch_accel_x'][start_fall_index:end_fall_index + 1].idxmax())  # end_fall_index 一次跌倒发生的结束点
#         # 选取峰值附近的数据，假设跌倒发生在2s内，就让max_id向左向右的100个数据注释不变。
#         j = start_fall_index
#         while (j < max_id - 50):
#             df['annotation_1'][j] = 0.0
#             j = j + 1
#         k = end_fall_index
#         while (k > max_id + 50):
#             df['annotation_1'][k] = 0.0
#             k = k - 1
#     else:
#         i = i + 1
plt.plot(df['ch_accel_x'])
plt.plot(df['annotation_1'],label='1')
plt.legend()
plt.show()
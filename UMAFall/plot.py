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

adl_file='E:/Fall/UMAFall/dataprocess/adl/UMAFall_Subject_02_ADL_Aplausing_1_2017-04-26_19-47-07.csv'
fall_file='E:/Fall/UMAFall/dataprocess/fall/UMAFall_Subject_02_Fall_backwardFall_1_2016-06-13_20-51-32.csv'
df = pd.read_csv(adl_file, header=32, sep=';')
print()
# df = pd.read_csv(adl_file, header=32, sep=';')
# df = df.loc[df[' Sensor Type'] == 0]  # Accelerometer = 0
# df = df.loc[df[' Sensor ID'] == 4]  # 4; ANKLE; SensorTag
# df.reset_index(drop=True, inplace=True)
# plt.plot(df[' X-Axis'])
# plt.plot(df[' Y-Axis'])
# plt.plot(df[' Z-Axis'])
# plt.show()
# fall_dir="E:/Fall/UMAFall/dataprocess/fall"
# file = glob.glob(os.path.join(fall_dir, "*.csv"))
# for f in file:
#     df = pd.read_csv(f, header=32, sep=';')
#     df = df.loc[df[' Sensor Type'] == 0]  # Accelerometer = 0
#     df = df.loc[df[' Sensor ID'] == 4]  # 4; ANKLE; SensorTag
#     df.reset_index(drop=True, inplace=True)
#     plt.plot(df[' X-Axis'])
#     plt.plot(df[' Y-Axis'])
#     plt.plot(df[' Z-Axis'])
#     plt.show()
import matplotlib.pyplot as plt
import pandas as pd
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
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
from sklearn.model_selection import KFold
from timeit import default_timer as timer
import csv
# step1 import 相关模块
# ES08 male 55
# ADL->Fall ->ADL 不管时间顺序
# step2:指定输入网络的训练集和测试集，如指定训练集的输入 x_train 和标签y_train，测试集的输入 x_test 和标签 y_test。
FEATURES =18
RANDOM_SEED = 42
TIME_STEPS=100 #100HZ
step=75


def m_plot(y_test_label, y_pred_label, test_index, segments, knum):
    loupan_index = []
    for i in range(len(y_test_label)):
        if (y_test_label[i] == 1 and y_pred_label[i] == 0):
            loupan_index.append(i)
    all_index = []
    for j in range(len(loupan_index)):
        all_index.append(test_index[loupan_index[j]])
    for k in range(len(loupan_index)):
        loupan = segments[all_index[k]]
        plt.plot(loupan[:,6:7],label='accel_x_r')
        plt.plot(loupan[:, 7:8], label='accel_y_r')
        plt.plot(loupan[:, 8:9], label='accel_z_r')
        plt.legend()
        plt.savefig('E:/Fall/TST/figure/loupan' + str(knum) + str(k + 1) + '.png', bbox_inches='tight')
        plt.close()


def Dataprocess(parent_dir,filepath):
    df = pd.read_csv(filepath, sep=',', header=None)
    df = df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 21, 22, 23]]
    if(parent_dir[-3:]=='adl'):
        df.insert(18, 'label_1', [0] * len(df))
    else:
        df.insert(18, 'label_1', [1] * len(df))
    df = pd.DataFrame(df.values,
                        columns=['fsr1_l', 'fsr2_l', 'fsr3_l', 'fsr1_r', 'fsr2_r', 'fsr3_r', 'accel_x_r', 'accel_y_r',
                                'accel_z_r', 'gyro_x_r', 'gyro_y_r', 'gyro_z_r', 'accel_x_l', 'accel_y_l',
                                'accel_z_l', 'gyro_x_l', 'gyro_y_l', 'gyro_z_l', 'label_1'])
    df.reset_index(drop=True, inplace=True)
    scale_columns = ['fsr1_l', 'fsr2_l', 'fsr3_l', 'fsr1_r', 'fsr2_r', 'fsr3_r', 'accel_x_r', 'accel_y_r', 'accel_z_r',
                     'gyro_x_r', 'gyro_y_r', 'gyro_z_r', 'accel_x_l', 'accel_y_l', 'accel_z_l', 'gyro_x_l', 'gyro_y_l',
                     'gyro_z_l']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(df[scale_columns])
    df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
    count = 0
    segments = []
    labels = []
    for i in range(0, len(df) - TIME_STEPS, step):
        fl1 = df['fsr1_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        fl2 = df['fsr2_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        fl3 = df['fsr3_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        fr1 = df['fsr1_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        fr2 = df['fsr2_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        fr3 = df['fsr3_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        accxr = df['accel_x_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        accyr = df['accel_y_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        acczr = df['accel_z_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        groxr = df['gyro_x_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        groyr = df['gyro_y_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        grozr = df['gyro_z_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
        accxl = df['accel_x_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        accyl = df['accel_y_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        acczl = df['accel_z_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        groxl = df['gyro_x_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        groyl = df['gyro_y_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        grozl = df['gyro_z_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
        segment = np.hstack((
                            fl1, fl2, fl3, fr1, fr2, fr3, accxr, accyr, acczr, groxr, groyr, grozr, accxl, accyl, acczl,
                            groxl, groyl, grozl))
        segments.append(segment)
        if (parent_dir[-3:] == 'adl'):
            labels.append(0)
        else:
            labels.append(1)
    return segments,labels

segments_all=[]
labels_all=[]
adl_file='E:/Fall/TST/ES02/adl'
fall_file='E:/Fall/TST/ES02/fall'
file = glob.glob(os.path.join(adl_file, "*.txt"))
df=pd.DataFrame()
for f in file:
    # filepath=adl_file+'/'+f
    s,l=Dataprocess(adl_file,f)
    segments_all = segments_all +s
    labels_all = labels_all +l

file = glob.glob(os.path.join(fall_file, "*.txt"))
for f in file:
    # filepath=fall_file+'/'+f
    s,l=Dataprocess(fall_file,f)
    segments_all = segments_all +s
    labels_all = labels_all +l
#------------------3折交叉验证-------------------------
floder = KFold(n_splits=3, random_state=42, shuffle=True)
reshaped_segments =np.asarray(segments_all,dtype=np.float32).reshape(-1,TIME_STEPS,FEATURES)
labels=np.asarray(pd.get_dummies(labels_all),dtype=np.float32)#是利用pandas实现one hot encode的方式。
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
for train_index, test_index in floder.split(reshaped_segments):
    k_fold_num +=1
    X_train, X_test = reshaped_segments[train_index], reshaped_segments[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #------------------------TCN--------------------------
    model_TCN = tf.keras.models.Sequential([
        TCN(input_shape=(X_train.shape[1],X_train.shape[2]),dilations=(1, 2,4,8,16)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    #---------------------model.compile--------------------------------------------------
    model_TCN.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='adam')
    if(k_fold_num==1):
        checkpoint_save_path1 ="E:/Fall/TST/checkpoint/TCN1.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)
    if(k_fold_num==2):
        checkpoint_save_path1 ="E:/Fall/TST/checkpoint/TCN2.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)
    if(k_fold_num==3):
        checkpoint_save_path1 = "E:/Fall/TST/checkpoint/TCN3.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path1,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
   # model_TCN.fit(X_train, y_train, epochs=12, batch_size=5, verbose=1,shuffle=True,callbacks=[model_checkpoint_callback])
    model_TCN.summary()
    loss1, acc1 = model_TCN.evaluate(X_test, y_test, verbose=1)
    LOSS.append(loss1)
    ACC.append(acc1)
    y_pred_label=np.argmax(model_TCN.predict(X_test), axis=1)
    y_test_label=np.argmax(y_test, axis=1)
    #混淆矩阵
    # C=confusion_matrix(y_test_label, y_pred_label)
    # sns.heatmap(C, annot=True)
    confusion_mat= confusion_matrix(y_test_label, y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=['ADL', 'Fall'])
    disp.plot(
        include_values=True,  # 混淆矩阵每个单元格上显示具体数值
        cmap=plt.cm.Blues,
        ax=None,  # 同上
        xticks_rotation="horizontal",  # 同上
        values_format="d"  # 显示的数值格式
    )
    title = "Confusion matrix"
    disp.ax_.set_title(title)
    if(k_fold_num==1):
        plt.savefig('E:/Fall/TST/figure/k1-TCN-C.png',bbox_inches='tight')
        plt.close()
        m_plot(y_test_label, y_pred_label, test_index,reshaped_segments, 1)
    if(k_fold_num==2):
        plt.savefig('E:/Fall/TST/figure/k2-TCN-C.png',bbox_inches='tight')
        plt.close()
        m_plot(y_test_label, y_pred_label, test_index, reshaped_segments, 2)
    if(k_fold_num==3):
        plt.savefig('E:/Fall/TST/figure/k3-TCN-C.png',bbox_inches='tight')
        plt.close()
        m_plot(y_test_label, y_pred_label, test_index, reshaped_segments, 3)
    plt.close()
    # tn:将ADL预测为ADL  tp:将跌倒预测为跌倒  fp:将ADL预测为跌倒 1 fn：将跌倒预测为ADL
    tn, fp, fn, tp = confusion_matrix(y_test_label, y_pred_label).ravel()
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)
    Accuracy.append((tp + tn) / (tp + tn + fp + fn))
    Precision.append(tp / (tp + fp))
    Recall.append(tp / (tp + fn))
for i in range(3):
    print('-------------'+str(i+1)+'-------------')

    print("K fold  TCN acc:{:4.4f}%" .format(100*ACC[i]))
    print("K fold average TCN loss: {:4.4f}%" .format(100*LOSS[i]))
    print("K fold tn:", TN[i])  # 查准率
    print("K fold fp:",  FP[i])  # 查全率
    print("K fold fn:", FN[i])  # 查准率
    print("K fold tp:",  TP[i])  # 查全率
    print("K fold Accuracy:", Accuracy[i])
    print("K fold precision:", Precision[i])  # 查准率
    print("K fold recall:",  Recall[i])  # 查全率
print('--------------------------------------------------')
print("K fold  TCN acc:{:4.4f}%".format(100 * ACC[i]))
print("K fold  TCN loss: {:4.4f}%".format(100 * LOSS[i]))
print("K fold average Accuracy:", np.array(Accuracy).mean())
print("K fold average precision:", np.array(Precision).mean())  # 查准率
print("K fold average recall:",np.array(Recall).mean())  # 查全率


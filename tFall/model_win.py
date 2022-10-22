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

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
"""
-adlV(2787,301)
    -adl_accx (929,301)
    -adl_accy (929,301)
    -adl_accz (929,301)
-fallV(153,301)
    -fall_accx (51,301)
    -fall_accy (51,301)
    -fall_accz (51,301)
    
    滑动窗口，要求感受野>滑动窗口大小
"""
adlV=np.loadtxt('E:/Fall/tFall/data/publicFallDetector201307/data201307/person0/adlProcessedVector/0adlPV.dat')
adl_accx=adlV[0::3,:] #contains the x-axis
adl_accy=adlV[1::3,:] #contains the y-axis
adl_accz=adlV[2::3,:] #contains the z-axis
fallV=np.loadtxt('E:/Fall/tFall/data/publicFallDetector201307/data201307/person0/fallProcessedVector/0fallPV.dat')
fall_accx=fallV[0::3,:]
fall_accy=fallV[1::3,:]
fall_accz=fallV[2::3,:]
segments=[]
labels=[]
adl_accx=adl_accx[0:201]
adl_accy=adl_accy[0:201]
adl_accz=adl_accz[0:201]
for i in range(len(adl_accx)):
    x=adl_accx[i].reshape(-1,1)
    y=adl_accy[i].reshape(-1,1)
    z=adl_accz[i].reshape(-1,1)
    all=np.hstack((x,y,z))
    segments.append(all)
    labels.append(0)
for i in range(len(fall_accx)):
    x=fall_accx[i].reshape(-1,1)
    y=fall_accy[i].reshape(-1,1)
    z=fall_accz[i].reshape(-1,1)
    all=np.hstack((x,y,z))
    segments.append(all)
    labels.append(1)
labels=np.asarray(pd.get_dummies(labels),dtype=np.float32)#是利用pandas实现one hot encode的方式。
reshaped_segments=np.asarray(segments)
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
floder = KFold(n_splits=3, random_state=42, shuffle=True)
for train_index, test_index in floder.split(reshaped_segments):
    k_fold_num +=1
    X_train, X_test = reshaped_segments[train_index], reshaped_segments[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #------------------------TCN--------------------------
    # 感受野大小 1+ nb_stacks*((kernel_size-1)*(1+2+4+..))
    model_TCN = tf.keras.models.Sequential([
        TCN(input_shape=(X_train.shape[1],X_train.shape[2]),dilations=(1, 2,4,8,16,32),kernel_size=6),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    #---------------------model.compile--------------------------------------------------
    model_TCN.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='adam')
    if(k_fold_num==1):
        checkpoint_save_path ="E:/Fall/tFall/checkpoint/tFall1.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    if(k_fold_num==2):
        checkpoint_save_path ="E:/Fall/tFall/checkpoint/tFall2.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    if(k_fold_num==3):
        checkpoint_save_path = "E:/Fall/tFall/checkpoint/tFall3.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
    model_TCN.fit(X_train, y_train, epochs=12, batch_size=16, verbose=1,shuffle=True,callbacks=[model_checkpoint_callback])
    model_TCN.summary()
    loss1, acc1 = model_TCN.evaluate(X_test, y_test, verbose=1)
    LOSS.append(loss1)
    ACC.append(acc1)
    predict=model_TCN.predict(X_test)
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
        plt.savefig('E:/Fall/tFall/figure/k1-TCN-C.png',bbox_inches='tight')
    if(k_fold_num==2):
        plt.savefig('E:/Fall/tFall/figure/k2-TCN-C.png',bbox_inches='tight')
    if(k_fold_num==3):
        plt.savefig('E:/Fall/tFall/figure/k3-TCN-C.png',bbox_inches='tight')
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
    # Labels = []
    # Labels_number = []
    # Fall_start = []
    # Fall_end = []
    # Labels, Labels_number, Fall_start, Fall_end = isFall(y_pred_label)
    # plt.subplot(2,2,1)
    # plt.plot(df['ch_accel_x'],label='ch_accel_x')
    # plt.plot(df['ch_accel_y'], label='ch_accel_y')
    # plt.plot(df['ch_accel_z'], label='ch_accel_z')
    # plt.legend()
    # plt.subplot(2,1,2)
    # plt.plot(df['ch_gyro_x'],label='ch_gyro_x')
    # plt.plot(df['ch_gyro_y'], label='ch_gyro_y')
    # plt.plot(df['ch_gyro_z'], label='ch_gyro_z')
    # plt.legend()
    # plt.subplot(3, 1, 3)
    # plt.title('UR')
    # plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
    # plt.scatter(test_index[Fall_end], np.array(len(Fall_end) * [1]), label="warning", c='red', s=18)
    # plt.legend()
    # if (k_fold_num == 1):
    #     #plt.figure(1,figsize=(20,10))
    #     plt.savefig('E:/Fall/UR/figure/k1-Warning.svg')
    # if (k_fold_num == 2):
    #     #plt.figure(2, figsize=(20, 10))
    #     plt.savefig('E:/Fall/UR/figure/k2-Warning.svg')
    # if (k_fold_num == 3):
    #    # plt.figure(3,figsize=(20,10))
    #     plt.savefig('E:/Fall/UR/figure/k3-Warning.svg')
    # plt.close()
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
# # plt.scatter(test_index,y_pred_label,s=1,c='red')
# # plt.scatter(test_index,y_test_label,s=1,c='green')
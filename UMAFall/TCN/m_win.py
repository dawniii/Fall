"""
data：2022.10.19
1s大概200个数据
"""
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
        plt.plot(loupan[:, 0:1], label=' X-Axis')
        plt.plot(loupan[:, 1:2], label=' Y-Axis')
        plt.plot(loupan[:, 2:3], label=' Z-Axis')
        plt.legend()
        plt.savefig('E:/Fall/UMAFall/figure/loupan' + str(knum) + str(k + 1) + '.png', bbox_inches='tight')
        plt.close()
FEATURES =3
RANDOM_SEED = 42
TIME_STEPS=200
step=150

#日常活动 0 跌倒 1
def Dataprocess(ParentDir,filepath):
    df = pd.read_csv(filepath, header=32, sep=';')
    df = df.loc[df[' Sensor Type'] == 0]# Accelerometer = 0
    df = df.loc[df[' Sensor ID'] == 4]# 4; ANKLE; SensorTag
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
    if ParentDir[-3:]=='adl':
        label = len(segment)*[0]
        # 构建一个segments长的列表，值全为0
    else:
        label = len(segment) * [1]
        #值全为1
    return segment,label
segments=[]
labels=[]
adl_dir="E:/Fall/UMAFall/dataprocess18/adl"
fall_dir="E:/Fall/UMAFall/dataprocess18/fall"
file = glob.glob(os.path.join(adl_dir, "*.csv"))
for f in file:
    segment = []
    label = []
    segment,label=Dataprocess(adl_dir,f)
    segments +=segment
    labels +=label
file = glob.glob(os.path.join(fall_dir, "*.csv"))
for f2 in file:
    segment = []
    label = []
    segment,label=Dataprocess(fall_dir,f2)
    segments +=segment
    labels +=label

#------------------3折交叉验证-------------------------
floder = KFold(n_splits=3, random_state=42, shuffle=True)
# floder = KFold(n_splits=3)
reshaped_segments =np.asarray(segments,dtype=np.float32).reshape(-1,TIME_STEPS,FEATURES)
labels=np.asarray(pd.get_dummies(labels),dtype=np.float32)#是利用pandas实现one hot encode的方式。
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
        TCN(input_shape=(X_train.shape[1],X_train.shape[2]),dilations=(1, 2,4,8,16),dropout_rate=0.2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    #---------------------model.compile--------------------------------------------------
    model_TCN.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='adam')
    if(k_fold_num==1):
        checkpoint_save_path1 ="E:/Fall/UMAFall/checkpoints/TCN1.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)
    if(k_fold_num==2):
        checkpoint_save_path1 ="E:/Fall/UMAFall/checkpoints/TCN2.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)
    if(k_fold_num==3):
        checkpoint_save_path1 = "E:/Fall/UMAFall/checkpoints/TCN3.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path1,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
    model_TCN.fit(X_train, y_train, epochs=12, batch_size=5, verbose=1,shuffle=True,callbacks=[model_checkpoint_callback])
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
        plt.savefig('E:/Fall/UMAFall/figure/k1-TCN-C.png',bbox_inches='tight')
        plt.close()
        m_plot(y_test_label, y_pred_label, test_index, segments, 1)
    if(k_fold_num==2):
        plt.savefig('E:/Fall/UMAFall/figure/k2-TCN-C.png',bbox_inches='tight')
        plt.close()
        m_plot(y_test_label, y_pred_label, test_index, segments, 2)
    if(k_fold_num==3):
        plt.savefig('E:/Fall/UMAFall/figure/k3-TCN-C.png',bbox_inches='tight')
        plt.close()
        m_plot(y_test_label, y_pred_label, test_index, segments, 3)
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
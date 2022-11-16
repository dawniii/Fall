import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tcn import TCN, tcn_full_summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import glob,os
from sklearn.model_selection import KFold
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
f='E:/Fall/UPFall/data/CompleteDataSet.csv'
df=pd.read_csv(f,header=None,skiprows=2)
df=df[[1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,25,26,27,29,30,31,32,33,34,43,44,45,46]]
columns=['Ankle_ACC_X','Ankle_ACC_Y','Ankle_ACC_Z','Ankle_ANG_X','Ankle_ANG_Y','Ankle_ANG_Z','RP_ACC_X','RP_ACC_Y','RP_ACC_Z','RP_ANG_X','RP_ANG_Y','RP_ANG_Z',
         'BELT_ACC_X', 'BELT_ACC_Y', 'BELT_ACC_Z', 'BELT_ANG_X', 'BELT_ANG_Y', 'BELT_ANG_Z','NECK_ACC_X','NECK_ACC_Y','NECK_ACC_Z','NECK_ANG_X','NECK_ANG_Y','NECK_ANG_Z','WRST_ACC_X','WRST_ACC_Y','WRST_ACC_Z','WRST_ANG_X','WRST_ANG_Y','WRST_ANG_Z',
         'Subject','Activity','Trial','Tag']
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
            scale_columns = ['Ankle_ACC_X','Ankle_ACC_Y','Ankle_ACC_Z','Ankle_ANG_X','Ankle_ANG_Y','Ankle_ANG_Z','RP_ACC_X','RP_ACC_Y','RP_ACC_Z','RP_ANG_X','RP_ANG_Y','RP_ANG_Z',
         'BELT_ACC_X', 'BELT_ACC_Y', 'BELT_ACC_Z', 'BELT_ANG_X', 'BELT_ANG_Y', 'BELT_ANG_Z','NECK_ACC_X','NECK_ACC_Y','NECK_ACC_Z','NECK_ANG_X','NECK_ANG_Y','NECK_ANG_Z','WRST_ACC_X','WRST_ACC_Y','WRST_ACC_Z','WRST_ANG_X','WRST_ANG_Y','WRST_ANG_Z']
            # 归一化，映射到-1到1之间
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(dk[scale_columns])
            dk.loc[:, scale_columns] = scaler.transform(dk[scale_columns].to_numpy())
            for i in range(0, len(dk) - TIME_STEPS, step):
                x1 = dk['Ankle_ACC_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y1 = dk['Ankle_ACC_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z1 = dk['Ankle_ACC_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x2 = dk['Ankle_ANG_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y2 = dk['Ankle_ANG_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z2 = dk['Ankle_ANG_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x3 = dk['RP_ACC_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y3 = dk['RP_ACC_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z3 = dk['RP_ACC_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x4 = dk['RP_ANG_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y4 = dk['RP_ANG_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z4 = dk['RP_ANG_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x5 = dk['BELT_ACC_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y5 = dk['BELT_ACC_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z5 = dk['BELT_ACC_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x6 = dk['BELT_ANG_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y6 = dk['BELT_ANG_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z6 = dk['BELT_ANG_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x7 = dk['NECK_ACC_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y7 = dk['NECK_ACC_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z7 = dk['NECK_ACC_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x8 = dk['NECK_ANG_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y8 = dk['NECK_ANG_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z8 = dk['NECK_ANG_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x9 = dk['WRST_ACC_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y9 = dk['WRST_ACC_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z9 = dk['WRST_ACC_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                x10 = dk['WRST_ANG_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
                y10 = dk['WRST_ANG_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
                z10 = dk['WRST_ANG_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
                segments.append(np.hstack((x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4,x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8,x9, y9, z9, x10, y10, z10)))
                #segments.append(np.hstack((x1, y1, z1, x2, y2, z2)))
                if(activity<5):
                    labels.append(1)
                else:
                    labels.append(0)
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
floder = KFold(n_splits=3, random_state=42, shuffle=True)
for train_index, test_index in floder.split(reshaped_segments):
    k_fold_num +=1
    X_train, X_test = reshaped_segments[train_index], reshaped_segments[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #------------------------TCN--------------------------
    # 感受野大小 1+ nb_stacks*((kernel_size-1)*(1+2+4+..))
    model_LSTM = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=64,input_shape=(X_train.shape[1],X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    #---------------------model.compile--------------------------------------------------
    model_LSTM.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='adam')
    if(k_fold_num==1):
        checkpoint_save_path = "/UPFall/LSTM/checkpoint/UPFall1.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_LSTM.load_weights(checkpoint_save_path)
    if(k_fold_num==2):
        checkpoint_save_path = "/UPFall/LSTM/checkpoint/UPFall2.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_LSTM.load_weights(checkpoint_save_path)
    if(k_fold_num==3):
        checkpoint_save_path = "/UPFall/LSTM/checkpoint/UPFall3.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_LSTM.load_weights(checkpoint_save_path)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
    model_LSTM.fit(X_train, y_train, epochs=12, batch_size=5, verbose=1,shuffle=True,callbacks=[model_checkpoint_callback])
    model_LSTM.summary()
    loss1, acc1 = model_LSTM.evaluate(X_test, y_test, verbose=1)
    LOSS.append(loss1)
    ACC.append(acc1)
    predict=model_LSTM.predict(X_test)
    y_pred_label=np.argmax(model_LSTM.predict(X_test), axis=1)
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
        plt.savefig('E:/Fall/UPFall/LSTM/figure/k1-C.png',bbox_inches='tight')
    if(k_fold_num==2):
        plt.savefig('E:/Fall/UPFall/LSTM/figure/k2-C.png',bbox_inches='tight')
    if(k_fold_num==3):
        plt.savefig('E:/Fall/UPFall/LSTM/figure/k3-C.png',bbox_inches='tight')
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
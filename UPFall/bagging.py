# bagging mlp ensemble on blobs dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
from tcn import TCN
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot, pyplot as plt
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import glob,os
import numpy as np
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

FEATURES =3
RANDOM_SEED = 42
TIME_STEPS=200
step=150

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

#(ndarray （50，200，3))
# evaluate a single mlp model
def evaluate_model(trainX, trainy, testX, testy):
	# encode targets
	trainy_enc = to_categorical(trainy)
	testy_enc = to_categorical(testy)
	# define model
	model = tf.keras.models.Sequential()
	model.add(TCN(input_shape=(trainX.shape[1],trainX.shape[2]),dilations=(1, 2,4,8,16),dropout_rate=0.2))
	model.add( tf.keras.layers.Dense(trainy.shape[1], activation='softmax'))
	#model.add( tf.keras.layers.Dense(3, activation='softmax'))
	model.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='adam')
	# fit model
	model.fit(trainX, trainy, epochs=12, batch_size=5, verbose=1, shuffle=True)
	#model.fit(trainX, trainy_enc, epochs=50, verbose=0)
	model.summary()
	# evaluate the model
	loss1, acc1 = model.evaluate(testX, testy, verbose=1)# loss1, acc1
	return model, acc1

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]#对于members中的每一个model遍历一遍做预测
	yhats = array(yhats)
	# sum across ensemble members
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)
	return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]#2个模型的集成、3个模型的集成、4个模型的集成...
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	testy = argmax(testy, axis=1)
	# calculate accuracy
	confusion_mat = confusion_matrix(testy,yhat)
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
	plt.savefig('E:/Fall/UPFall/figure2/'+str(n_members)+'-TCN-C.png', bbox_inches='tight')
	plt.close()
	return accuracy_score(testy, yhat)

# generate 2d classification dataset
#dataX, datay = make_blobs(n_samples=55000, centers=3, n_features=2, cluster_std=2, random_state=2)

# X, newX = [:32, :], reshaped_segments[32:, :]
# y, newy = labels[:32], labels[32:]
X,newX,y,newy=train_test_split(reshaped_segments,labels,test_size=0.36,random_state=42)
"""X和y是训练集，newX和newy是测试集"""
# multiple train-test splits
n_splits = 10
scores, members = list(), list()
for _ in range(n_splits):
	# select indexes
	ix = [i for i in range(len(X))]#索引
	train_ix = resample(ix, replace=True, n_samples=422)#ix为[0,4999] resample为带有替换的子样本，样本大小为4500（640-》576）,即数据的90%
	test_ix = [x for x in ix if x not in train_ix]#没在test_ix中的索引
	# select data
	trainX, trainy = X[train_ix], y[train_ix]#train_ix为list4500,test_ix为list2034
	testX, testy = X[test_ix], y[test_ix]
	# evaluate model
	model, test_acc = evaluate_model(trainX, trainy, testX, testy)
	print('>%.3f' % test_acc)
	scores.append(test_acc)
	members.append(model) #添加的是用4500个样本单独进行训练的模型
# summarize expected performance
print('Estimated Accuracy %.3f (%.3f)' % (mean(scores), std(scores)))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, n_splits+1):
	ensemble_score = evaluate_n_members(members, i, newX, newy)#members是10个Model的列表
	#newy_enc = to_categorical(newy)
	newy_enc = newy
	_, single_score = members[i-1].evaluate(newX, newy_enc, verbose=0)#单个模型 newX是测试集
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# plot score vs number of ensemble members
print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
x_axis = [i for i in range(1, n_splits+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
plt.savefig('E:/Fall/UPFall/figure2/kk.png', bbox_inches='tight')
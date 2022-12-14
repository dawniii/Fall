# bagging mlp ensemble on blobs dataset
from scipy.stats import stats
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

TIME_STEPS=100 #100HZ
FEATURES =12
RANDOM_SEED = 42
step=75
# file='E:/Fall/Cogent/dataprocess/falls/subject_1'
# df=pd.read_csv(file)
# df.loc[df['annotation_1'].isin([2]),'annotation_1']=0
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
        plt.plot(loupan[:,0:1],label='ch_accel_x')
        plt.plot(loupan[:,1:2], label='ch_accel_y')
        plt.plot(loupan[:,2:3], label='ch_accel_z')
        plt.legend()
        plt.savefig('E:/Fall/Cogent/figure2/loupan' + str(knum) + str(k + 1) + '.png', bbox_inches='tight')
        plt.close()

def DataProcess(filepath):
    segments = []
    labels = []
    labels_number = []
    df = pd.read_csv(filepath)
    df.loc[df['annotation_1'].isin([2]), 'annotation_1'] = 0
  #  df=df[0:100000]
    # df['annotation_1'] = df['annotation_1'].replace(2, 1)
    # ??????annotation_1???????????????2??????
    # df = df[~df['annotation_1'].isin([2])]
    # ??????df?????????
    # df.reset_index(drop=True, inplace=True)
    i = 0
    start_list = []  # ??????????????????????????????
    end_list = []  # ???????????????????????????
    while (i < len(df)):
        if (df['annotation_1'][i] == 1.0):  # ???????????????????????????
            start_fall_index = i  # ??????????????????????????????
            end = i + 1
            while ((0.0 in list(df['annotation_1'][start_fall_index:end])) == False):
                i = i + 1
                end = i
            end_fall_index = end - 2
            start_list.append(start_fall_index)
            end_list.append(end_fall_index)
            # ???????????????????????????
            max_id = df['ch_accel_x'][start_fall_index:end_fall_index + 1].idxmax()
            print(start_fall_index, end_fall_index, df['ch_accel_x'][start_fall_index:end_fall_index + 1].max(),
                  df['ch_accel_x'][start_fall_index:end_fall_index + 1].idxmax())  # end_fall_index ??????????????????????????????
            # ???????????????????????????????????????????????????2s????????????max_id???????????????100????????????????????????
            j = start_fall_index
            while (j < max_id - 50):
                df['annotation_1'][j] = 0.0
                j = j + 1
            k = end_fall_index
            while (k > max_id + 50):
                df['annotation_1'][k] = 0.0
                k = k - 1
        else:
            i = i + 1
    scale_columns = ['ch_accel_x', 'ch_accel_y', 'ch_accel_z', 'ch_gyro_x', 'ch_gyro_y', 'ch_gyro_z', 'th_accel_x',
                     'th_accel_y', 'th_accel_z', 'th_gyro_x', 'th_gyro_y', 'th_gyro_z']
    # ?????????????????????-1???1??????
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(df[scale_columns])
    df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
    for i in range(0, len(df) - TIME_STEPS, step):
        x = df['ch_accel_x'].values[i:i + TIME_STEPS].reshape(-1, 1)
        y = df['ch_accel_y'].values[i:i + TIME_STEPS].reshape(-1, 1)
        z = df['ch_accel_z'].values[i:i + TIME_STEPS].reshape(-1, 1)
        xs = df['ch_gyro_x'].values[i:i + TIME_STEPS].reshape(-1, 1)
        ys = df['ch_gyro_y'].values[i:i + TIME_STEPS].reshape(-1, 1)
        zs = df['ch_gyro_z'].values[i:i + TIME_STEPS].reshape(-1, 1)
        xx = df['th_accel_x'].values[i:i + TIME_STEPS].reshape(-1, 1)
        yy = df['th_accel_y'].values[i:i + TIME_STEPS].reshape(-1, 1)
        zz = df['th_accel_z'].values[i:i + TIME_STEPS].reshape(-1, 1)
        xxs = df['th_gyro_x'].values[i:i + TIME_STEPS].reshape(-1, 1)
        yys = df['th_gyro_y'].values[i:i + TIME_STEPS].reshape(-1, 1)
        zzs = df['th_gyro_z'].values[i:i + TIME_STEPS].reshape(-1, 1)
        label = stats.mode(df['annotation_1'][i:i + TIME_STEPS])[0][0]  # ?????????????????????
        label_number = stats.mode(df['annotation_1'][i:i + TIME_STEPS])[1][0]  # ??????????????????????????????
        segments.append(np.hstack((x, y, z, xs, ys, zs, xx, yy, zz, xxs, yys, zzs)))
        labels.append(label)
        labels_number.append(label_number)
    return segments,labels

segments_all=[]
labels_all=[]
ParentDir='E:/Fall/Cogent/dataprocess/falls'
file=os.listdir(ParentDir)
for f in file:
    filepath=ParentDir+'/'+f
    s,l=DataProcess(filepath)
    segments_all = segments_all +s
    labels_all = labels_all +l

#??????labels_all[i]???0???index
new_labels_all=[]
new_segments_all=[]
adl_num=0
fall_num=0
for k in range(len(labels_all)):
    if(labels_all[k]==0 and adl_num<=220):
        new_segments_all.append(segments_all[k])
        new_labels_all.append(labels_all[k])
        adl_num +=1
    elif(labels_all[k]==1):
        new_segments_all.append(segments_all[k])
        new_labels_all.append(labels_all[k])
        fall_num +=1

reshaped_segments =np.asarray(new_segments_all,dtype=np.float32).reshape(-1,TIME_STEPS,FEATURES)
labels=np.asarray(pd.get_dummies(new_labels_all),dtype=np.float32)#?????????pandas??????one hot encode????????????

#(ndarray ???50???200???3))
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
	yhats = [model.predict(testX) for model in members]#??????members???????????????model?????????????????????
	yhats = array(yhats)
	# sum across ensemble members
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)
	return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]#2?????????????????????3?????????????????????4??????????????????...
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	testy = argmax(testy, axis=1)
	# calculate accuracy
	confusion_mat = confusion_matrix(testy,yhat)
	disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=['ADL', 'Fall'])
	disp.plot(
		include_values=True,  # ????????????????????????????????????????????????
		cmap=plt.cm.Blues,
		ax=None,  # ??????
		xticks_rotation="horizontal",  # ??????
		values_format="d"  # ?????????????????????
	)
	title = "Confusion matrix"
	disp.ax_.set_title(title)
	plt.savefig('E:/Fall/Cogent/figure3/'+str(n_members)+'-TCN-C.png', bbox_inches='tight')
	plt.close()
	return accuracy_score(testy, yhat)

# generate 2d classification dataset
#dataX, datay = make_blobs(n_samples=55000, centers=3, n_features=2, cluster_std=2, random_state=2)

# X, newX = [:32, :], reshaped_segments[32:, :]
# y, newy = labels[:32], labels[32:]
X,newX,y,newy=train_test_split(reshaped_segments,labels,test_size=0.36,random_state=42)
"""X???y???????????????newX???newy????????????"""
# multiple train-test splits
n_splits = 10
scores, members = list(), list()
for _ in range(n_splits):
	# select indexes
	ix = [i for i in range(len(X))]#??????
	train_ix = resample(ix, replace=True, n_samples=138)#ix???[0,4999] resample?????????????????????????????????????????????4500???153-???138???,????????????90%
	test_ix = [x for x in ix if x not in train_ix]#??????test_ix????????????
	# select data
	trainX, trainy = X[train_ix], y[train_ix]#train_ix???list4500,test_ix???list2034
	testX, testy = X[test_ix], y[test_ix]
	# evaluate model
	model, test_acc = evaluate_model(trainX, trainy, testX, testy)
	print('>%.3f' % test_acc)
	scores.append(test_acc)
	members.append(model) #???????????????4500????????????????????????????????????
# summarize expected performance
print('Estimated Accuracy %.3f (%.3f)' % (mean(scores), std(scores)))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, n_splits+1):
	ensemble_score = evaluate_n_members(members, i, newX, newy)#members???10???Model?????????
	#newy_enc = to_categorical(newy)
	newy_enc = newy
	_, single_score = members[i-1].evaluate(newX, newy_enc, verbose=0)#???????????? newX????????????
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# plot score vs number of ensemble members
print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
x_axis = [i for i in range(1, n_splits+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
plt.savefig('E:/Fall/Cogent/kk.png', bbox_inches='tight')
plt.close()
import numpy as np
import keras
from random import sample
from keras.callbacks import History
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Reshape, Activation, MaxPooling2D, Input, SpatialDropout2D, Permute, Dropout, AveragePooling2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from sklearn.model_selection import train_test_split, StratifiedKFold
from matplotlib import pyplot as plt
results = open('person1.txt', 'w')
badNums = [43, 88, 89, 92, 100, 104]
#Xall =  np.zeros((4635, 64,321))
Xall =  np.zeros((4635, 64,100))#50 resampling
#Xall =  np.zeros((4635, 64,60))#30 resampling
#Xall =  np.zeros((4635, 64,20))#10 resampling
Yall = np.zeros((4635,1))

depthXIndex = 0
depthYIndex = 0
lossOutFile = open('lossfile.txt', 'w')
newallY = []
for personNum in range(1,110):
	if personNum in badNums:
		continue
	tmin, tmax = -1., 1.
	event_id = dict(hands=2, feet=3)
	subject = personNum
	runs = [6, 10, 14]  # motor imagery: hands vs feet
	raw_fnames = eegbci.load_data(subject, runs)
	raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
	raw.rename_channels(lambda x: x.strip('.'))
	events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
	epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

	epochs = epochs.copy().resample(50, npad='auto')#try 50,30,10

	X = epochs._data * 1000
	
	for i in range(0, X.shape[0]):
		Xall[depthXIndex] = X[i]
		depthXIndex+=1
	Y = events[:,-1] - 2
	for e in Y:
		newallY.append(e)
	for i in range(0, Y.shape[0]):
		Yall[depthYIndex] = Y[i]
		depthYIndex+=1
Xall = (Xall-np.mean(Xall)) / np.std(Xall)	#normalize data
loss = []
valLoss = []
acc = []
valAcc = []
random_seed = 1017
np.random.seed(random_seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
cvscores = []
histArr = []
numDone = 0
for train, test in kfold.split(Xall, Yall):
	input1 = Input(shape = (Xall[train].shape[1], Xall[train].shape[2],1))
	Xall2 = Xall[train].reshape(Xall[train].shape[0], Xall[train].shape[1], Xall[train].shape[2],1)	
	kernLength = 80
	dropoutRate = .2
	nbClasses = 2
	normRate = .25
	F1 = 8
	block1 = (Conv2D(F1, (1, kernLength), padding='same', input_shape = (Xall[train].shape[1], Xall[train].shape[0],1), use_bias=False))(input1)
	block1 = BatchNormalization(axis=1)(block1)
	block1 = DepthwiseConv2D((Xall.shape[1],1), use_bias=False, padding='same', depth_multiplier = 2, depthwise_constraint=max_norm(1.), data_format='channels_first')(block1)
	block1 = BatchNormalization(axis=1)(block1)
	block1 = Activation('elu')(block1)
	block1 = AveragePooling2D((1,4), data_format='channels_first')(block1)
	block1 = Dropout(dropoutRate)(block1)
	block2 = SeparableConv2D(16, (1,16), use_bias=False, padding='same',data_format='channels_first')(block1)
	block2 = BatchNormalization(axis=1)(block2)
	block2 = Activation('elu')(block2)
	block2 = AveragePooling2D((1,8), data_format='channels_first', padding='same')(block2)
	block2 = Dropout(dropoutRate)(block2)
	flatten = Flatten(name='flatten')(block2)
	dense = Dense(nbClasses, name='dense', kernel_constraint=max_norm(normRate))(flatten)
	softmax = Activation('softmax', name='softmax')(dense)
	model = Model(inputs=input1, outputs=softmax)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	YallCat = to_categorical(Yall[train])
	Xall2test = Xall[test].reshape(Xall[test].shape[0], Xall[test].shape[1], Xall[test].shape[2],1)	
	history = model.fit(Xall2,YallCat, batch_size=16, epochs=25, verbose=1)
	histArr.append(history)
	scores = model.evaluate(Xall2test, to_categorical(Yall[test]), batch_size=2)
	cvscores.append(scores[1] * 100)
	numDone += 1
	print('number of rounds done out of 10:', numDone)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

showPlots = True
print(history.history)
accOutFile = open('accfile.txt', 'w')
for entry in histArr:
	for hentry in entry.history['loss']:
		lossOutFile.write(str(hentry) + '\t')
	lossOutFile.write('\n')
	for hentry in entry.history['accuracy']:
		accOutFile.write(str(hentry) + '\t')
	accOutFile.write('\n')
lossOutFile.close()
accOutFile.close()
results.close()

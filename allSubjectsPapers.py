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
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
badNums = [43, 88, 89, 92, 100, 104]
#Xall =  np.zeros((4635, 64,321))
#Xall =  np.zeros((4635, 64,100))#50 resampling
Xall =  np.zeros((4635, 64,60))#30 resampling
#Xall =  np.zeros((4635, 64,20))#10 resampling
Yall = np.zeros((4635,2))

depthXIndex = 0
depthYIndex = 0
outFile = open('personNum.txt', 'w')
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

	epochs = epochs.copy().resample(30, npad='auto')#try 50,30,10

	X = epochs._data * 1000
	
	for i in range(0, X.shape[0]):
		Xall[depthXIndex] = X[i]
		depthXIndex+=1
	Y = events[:,-1] - 2
	Y = to_categorical(Y)
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
(X_train, X_test, y_train, y_test) = train_test_split(Xall, Yall, test_size = .2, random_state=random_seed)
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size = .25, random_state=random_seed)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2], 1)
input1 = Input(shape = (X_train.shape[1], X_train.shape[2], 1))
		
kernLength = 80
dropoutRate = .2
nbClasses = 2
normRate = .25
F1 = 8
block1 = (Conv2D(F1, (1, kernLength), padding='same', input_shape = (X_train.shape[1], X_train.shape[0], 1), use_bias=False))(input1)
block1 = BatchNormalization(axis=1)(block1)
block1 = DepthwiseConv2D((X_train.shape[1],1), use_bias=False, padding='same', depth_multiplier = 2, depthwise_constraint=max_norm(1.), data_format='channels_first')(block1)
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
model.summary()
history = model.fit(X_train, y_train, batch_size=16, epochs=25, verbose=1, validation_data=(X_val, y_val),  shuffle=True)
probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds==y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))
score, acc = model.evaluate(X_test, y_test, batch_size=2)
print(model.metrics_names)
print('Test loss:', score)
print('Test accuracy:', acc)
showPlots = True
print('Training loss:', history.history['loss'][-1], 'Training accuracy:', history.history['accuracy'][-1], 'Validation loss:', history.history['val_loss'][-1], 'Validation accuracy:', history.history['val_accuracy'][-1])
if showPlots:
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'],loc='upper left')
	plt.show()
	plt.semilogy(history.history['loss'])
	plt.semilogy(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
results.close()
outFile.close()

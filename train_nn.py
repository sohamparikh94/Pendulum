import os
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from IPython import embed
from keras import optimizers
import matplotlib.pyplot as plt

from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_files",type=int, default=10)
	parser.add_argument("--num_layers", type=int)
	parser.add_argument("--num_units",nargs='+', type=int)
	parser.add_argument("--act",choices=['sigmoid','relu','tanh'],default='relu')
	parser.add_argument("--optimizer",choices=['sgd','rmsprop','adagrad','adam'],default='adam')
	parser.add_argument("--learning_rate",type=float,default=0.01)
	parser.add_argument("--batch_size",type=int,default=32)
	parser.add_argument("--num_epochs", type=int,default=10)
	parser.add_argument("--loss_function",type=str, default='mse')
	parser.add_argument("--save_path",type=str,default='model')
	args = parser.parse_args()
	return args

def prepare_data(folder,num_files):
	X = np.array([])
	Y = np.array([])
	for i in tqdm(range(num_files)):
		with open(os.path.join(folder,'raw_'+str(i)+'.pkl'),'rb') as f:
			data = np.array(pkl.load(f))
			data = np.reshape(data, [500*100,2])
			np.random.shuffle(data)
			X = np.append(X, data[:,0])
			Y = np.append(Y, data[:,1])
	X = np.array([x.tolist() for x in X.tolist()])
	Y = np.array([y.tolist() for y in Y.tolist()])
	cosX = X[:,0]
	sinX = X[:,1]
	""" The pendulum environment does not give theta as an output so we compute it below. It is between -pi and pi"""
	thetaX = np.zeros(np.shape(cosX))
	thetaX = np.multiply((cosX > 0).astype(int),(sinX > 0).astype(int))*(np.arcsin(sinX)) + np.multiply((cosX < 0).astype(int),(sinX > 0).astype(int))*(np.pi - np.arcsin(sinX)) + np.multiply((cosX > 0).astype(int),(sinX < 0).astype(int))*(np.arcsin(sinX)) + np.multiply((cosX < 0).astype(int),(sinX < 0).astype(int))*(-np.pi - np.arcsin(sinX))
	cosY = Y[:,0]
	sinY = Y[:,1]
	thetaY = np.zeros(np.shape(cosY))
	thetaY = np.multiply((cosY > 0).astype(int),(sinY > 0).astype(int))*(np.arcsin(sinY)) + np.multiply((cosY < 0).astype(int),(sinY > 0).astype(int))*(np.pi - np.arcsin(sinY)) + np.multiply((cosY > 0).astype(int),(sinY < 0).astype(int))*(np.arcsin(sinY)) + np.multiply((cosY < 0).astype(int),(sinY < 0).astype(int))*(-np.pi - np.arcsin(sinY))

	""" We use sigmoid activation in the last layer and hence, we need to set the outputs to be between 0 and 1"""

	Y[:,0] = (Y[:,0] - np.min(Y[:,0]))/(np.max(Y[:,0]) - np.min(Y[:,0]))
	Y[:,1] = (Y[:,1] - np.min(Y[:,1]))/(np.max(Y[:,1]) - np.min(Y[:,1]))
	Y[:,2] = (Y[:,2] - np.min(Y[:,2]))/(np.max(Y[:,2]) - np.min(Y[:,2]))
	thetaY = (thetaY - np.min(thetaY))/(np.max(thetaY) - np.min(thetaY))
	X = np.append(X,np.reshape(thetaX,[len(thetaX),1]),axis=1)
	Y = np.append(Y,np.reshape(thetaY,[len(thetaY),1]),axis=1)
	pkl.dump(X,open('data/data_X.pkl','wb'))
	pkl.dump(Y,open('data/data_X.pkl','wb'))
	return (X,Y)

def build_model(num_layers, num_units, act, optimizer, learning_rate, loss_function):
	model = Sequential()
	model.add(Dense(units=num_units[0],activation=act,input_dim=5,kernel_initializer='he_normal'))
	for layer in range(num_layers-1):
		model.add(Dense(units=num_units[layer+1],activation=act,kernel_initializer='he_normal'))
	model.add(Dense(units=4,activation='sigmoid',kernel_initializer='he_normal'))
	if(optimizer == 'sgd'):
		optimizer_fn = optimizers.SGD(lr=learning_rate)
	elif(optimizer=='rmsprop'):
		optimizer_fn = optimizers.RMSprop(lr=learning_rate)
	elif(optimizer=='adam'):
		optimizer_fn = optimizers.Adam(lr=learning_rate)
	else:
		optimizer_fn = optimizers.Adagrad(lr=learning_rate)
	model.compile(loss=loss_function,optimizer=optimizer_fn)

	return model

def train_model(model, X_train, Y_train, X_valid, Y_valid, num_epochs, batch_size, save_path):

	checkpoint = ModelCheckpoint(os.path.join(save_path,'model.hdf5'), monitor='val_acc', verbose=1)

	callbacks_list = [checkpoint]

	history = model.fit(X_train,Y_train,batch_size=batch_size,epochs=num_epochs,validation_data=(X_valid,Y_valid),callbacks = callbacks_list)
	return history


def main():
	args = parse_arguments()
	X,Y = prepare_data('data',args.num_files)
	n = len(X)
	X_train = X[:int(0.6*n)]
	Y_train = Y[:int(0.6*n)]
	X_valid = X[int(0.6*n):int(0.8*n)]
	Y_valid = Y[int(0.6*n):int(0.8*n)]
	X_test = X[int(0.8*n):]
	Y_test = Y[int(0.8*n):]
	del X
	del Y


	model = build_model(args.num_layers, args.num_units, args.act,args.optimizer, args.learning_rate, args.loss_function)
	
	history = train_model(model, X_train, Y_train, X_valid, Y_valid, args.num_epochs, args.batch_size, args.save_path)


	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig('loss_plot_' + str(args.learning_rate) + str(n) + '.png')


if __name__ == "__main__":
	main()


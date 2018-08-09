import numpy as np
import keras.models
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras import optimizers
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from Params import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# Test Tool
def Print(vec, file):
	with open(file, 'w') as fs:
		for val in vec:
			fs.write(str(val) + '\n')

class AutoEncoder:
	def __init__(self):
		self.model = Sequential()
		self.model.add(Dropout(rate=0.2, input_shape=(USER_NUM,)))
		self.model.add(Dense(LATENT_DIM,
			activation='sigmoid', use_bias=True,
			kernel_regularizer=None if REG_WEIGHT==0.0 else regularizers.l2(REG_WEIGHT)))
		self.model.add(Dense(USER_NUM, use_bias=True,
			kernel_regularizer=None if REG_WEIGHT==0.0 else regularizers.l2(REG_WEIGHT)))
		self.model.compile(optimizer=optimizers.Adam(lr=LR, decay=DECAY),
			loss=self.loss, metrics=[self.RMSE])
		# used to be rmsprop

	def loadModel(self):
		self.model = keras.models.load_model(LOAD_MODEL, {'loss': self.loss, 'RMSE': self.RMSE})

	def train(self, generator, validation_data=None):
		ret = self.model.fit_generator(generator, epochs=EPOCH, 
			workers=4, validation_data=validation_data, shuffle=True,
			steps_per_epoch=MOVIE_NUM/BATCH_SIZE, validation_steps=64)
		tem = 'Models/' + SAVE_PATH
		self.model.save(tem)
		jsonModel = self.model.to_json()
		with open(tem + '.arch', 'w') as fs:
			fs.write(jsonModel)
		self.model.save_weights(tem + '.weight')
		return ret

	def trainSimp(self, data, label, validation_data):
		ret = self.model.fit(x=data, y=label, batch_size=BATCH_SIZE, epochs=EPOCH,
			validation_data=validation_data, shuffle=True)
		tem = 'Models/' + SAVE_PATH
		self.model.save(tem)
		jsonModel = self.model.to_json()
		with open(tem + '.arch', 'w') as fs:
			fs.write(jsonModel)
		self.model.save_weights(tem + '.weight')
		return ret

	def evaluateSimp(self, data, label):
		return self.model.evaluate(x=data, y=label)

	def evaluate(self, generator):
		return self.model.evaluate_generator(generator, steps=10)

	def predict(self, x):
		return self.model.predict(x)

	def loss(self, y_true, y_pred):
		tem = tf.sign(y_true)
		nums = tf.reduce_sum(tem, axis=-1)
		nums = nums + (0.000001 - 0.000001 * tf.sign(nums)) # in case nums == 0
		return tf.reduce_sum(K.square(y_true - tf.multiply(y_pred, tem)), -1) #/ nums

	def RMSE(self, y_true, y_pred):
		tem = tf.sign(y_true)
		nums = tf.reduce_sum(tem)
		nums = nums + (0.000001 - 0.000001 * tf.sign(nums))
		temPred = (1 - tf.abs(tf.sign(y_pred))) * 3 # make avg prediction for 0 preds(IS IT POSSIBLE THAT THE NET GIVE IT SCORE ZERO???)
		temPred = y_pred + temPred
		return tf.sqrt(tf.reduce_sum(K.square((y_true - temPred) * tem)) / nums)


	# def loss(self, y_true, y_pred):
	# 	pos = tf.cast(tf.greater(y_true, 0.0), tf.float32)
	# 	preLoss = K.square((y_true - y_pred) * pos)

	# 	colLoss = (y_pred - tf.floor(y_pred)) * (tf.ceil(y_pred) - y_pred) * (1 - pos)

	# 	return tf.reduce_sum(preLoss + colLoss * COL_WEIGHT, -1)

	# def RMSE(self, y_true, y_pred):
	# 	tem = tf.cast(tf.greater(y_true, 0.0), tf.float32)
	# 	nums = tf.reduce_sum(tem)
	# 	nums = nums + (0.000001 - 0.000001 * tf.sign(nums))
	# 	temPred = (1 - tf.abs(tf.sign(y_pred))) * 3 # make avg prediction for 0 preds(IS IT POSSIBLE THAT THE NET GIVE IT SCORE ZERO???)
	# 	temPred = y_pred + temPred
	# 	return tf.sqrt(tf.reduce_sum(K.square((tf.abs(y_true) - temPred) * tem)) / nums)

import tensorflow as tf
from Params import *
from ToolScripts.TimeLogger import log
import numpy as np
from MakeData import ScipyMatMaker
import pickle

class AttentiveAE:
	def __init__(self, sess, trainMat, cvMat, testMat):
		self.sess = sess
		self.trainMat = trainMat
		self.trainMask = trainMat != 0
		self.trainNum = np.sum(self.trainMask)
		self.cvMat = cvMat
		self.cvMask = cvMat != 0
		self.cvNum = np.sum(self.cvMask)
		self.testMat = testMat
		self.testMask = testMat != 0
		self.testNum = np.sum(self.testMask)
		self.train_losses = list()
		self.train_rmse = list()
		self.test_lossses = list()
		self.test_RMSEs = list()
		print('hehe', self.trainNum, self.testNum, self.cvNum)
		exit()

	def run(self):
		self.prepare_model()
		log('Model Prepared')
		init = tf.global_variables_initializer()
		self.sess.run(init)
		log('Variables Inited')
		for ep in range(EPOCH):
			log('Epoch %d' % ep)
			self.train_model(ep)
			self.test_model(ep, self.cvMat, self.cvMask, self.cvNum)
			log('Epoch %d/%d: CvLoss = %f, CvRMSE = %f' \
				% (ep, EPOCH, self.test_lossses[-1], self.test_RMSEs[-1]))
		self.test_model(ep, self.testMat, self.testMask, self.testNum)
		log('Overall: TestLoss = %f, TestRMSE = %f' \
			% (self.test_lossses[-1], self.test_RMSEs[-1]))
		self.save_history()

	def original(self, V, W, mu, b):
		self.encoder = tf.nn.sigmoid(tf.matmul(self.input_R, V) + mu)
		self.decoder = tf.identity(tf.matmul(self.encoder, tf.transpose(W)) + b)

	def attentive(self, V, W, mu, b):
		reshaped_mask = tf.tile(tf.expand_dims(self.mask, 1), (1, USER_NUM, 1))
		reshaped_input = tf.tile(tf.expand_dims(self.input_R, 1), (1, USER_NUM, 1))
		curBatchSize = tf.shape(self.input_R)[0]
		reshaped_V = tf.tile(tf.expand_dims(V, 0), (curBatchSize, 1, 1))
		with tf.device('/gpu:0'):
			sim = tf.matmul(V, tf.transpose(V)) * reshaped_mask
			# sim = tf.ones((BATCH_SIZE, USER_NUM, USER_NUM))
			exp = tf.exp(sim)
			attention = exp / tf.reduce_sum(exp, axis=-1, keepdims=True)
		# attention = tf.nn.softmax(sim, axis=-1)
		self.encoder = tf.nn.sigmoid(tf.matmul(attention * reshaped_input, reshaped_V) + mu)
		self.decoder = tf.identity(tf.reduce_sum(self.encoder * W, -1) + b)

	def prepare_model(self):
		self.input_R = tf.placeholder(dtype=tf.float32, shape=[None, USER_NUM],
			name='input_R')
		self.mask = tf.placeholder(dtype=tf.float32, shape=[None, USER_NUM],
			name='mask')
		self.label = tf.placeholder(dtype=tf.float32, shape=[None, USER_NUM],
			name='label')

		V = tf.get_variable(name='V', initializer=tf.truncated_normal(
			shape=[USER_NUM, LATENT_DIM],
			mean=0, stddev=0.03), dtype=tf.float32)
		W = tf.get_variable(name='W', initializer=tf.truncated_normal(
			shape=[USER_NUM, LATENT_DIM],
			mean=0, stddev=0.03), dtype=tf.float32)
		mu = tf.get_variable(name='mu', initializer=tf.zeros(shape=LATENT_DIM),
			dtype=tf.float32)
		b = tf.get_variable(name='b', initializer=tf.zeros(shape=USER_NUM),
			dtype=tf.float32)

		self.original(V, W, mu, b)

		pre_loss = tf.reduce_sum(tf.square(self.decoder - self.label) * self.mask) #/ tf.reduce_sum(tf.cast(self.mask, tf.float32))
		reg_loss = tf.reduce_sum(tf.square(W) + tf.square(V))
		self.loss = pre_loss + REG_WEIGHT * reg_loss * tf.cast(tf.shape(self.input_R)[0], tf.float32)

		global_step = tf.Variable(0, trainable=False)
		decay_step = 10 * MOVIE_NUM / BATCH_SIZE
		learning_rate = tf.train.exponential_decay(LR, global_step, decay_step,
			0.96, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
			global_step=global_step)

	def train_model(self, itr):
		shuffledIds = np.random.permutation(MOVIE_NUM)
		epoch_loss = 0
		epoch_rmse = 0
		steps = int(np.ceil(MOVIE_NUM / BATCH_SIZE))
		for i in range(steps):
			ed = min((i+1) * BATCH_SIZE, MOVIE_NUM)
			batch_ids = shuffledIds[i * BATCH_SIZE: ed]
			_, loss, decoder = self.sess.run(
				[self.optimizer, self.loss, self.decoder],
				feed_dict={self.input_R: self.trainMat[batch_ids],
						   self.mask: self.trainMask[batch_ids],
						   self.label: self.trainMat[batch_ids]})
			epoch_loss += loss
			epoch_rmse += self.RMSE(decoder, self.trainMat[batch_ids],
				self.trainMask[batch_ids])
			log('Step %d/%d: loss = %f' % (i, steps, loss), oneline=True)
		self.train_losses.append(epoch_loss)
		avg_rmse = epoch_rmse / self.trainNum
		self.train_rmse.append(avg_rmse)
		log('Epoch %d/%d: loss = %f, rmse = %f' % (itr, EPOCH, epoch_loss, avg_rmse))

	def test_model(self, itr, label, mask, num):
		loss, decoder = self.sess.run(
			[self.loss, self.decoder],
			feed_dict={self.input_R: self.trainMat,
					   self.mask: mask,
					   self.label: label})
		rmse = self.RMSE(decoder, label, mask)
		self.test_lossses.append(loss)
		self.test_RMSEs.append(rmse / num)

	def RMSE(self, decoder, label, mask):
		return np.sum(np.square((decoder - label) * mask))

	def save_history(self):
		history = dict()
		history['loss'] = self.train_losses
		history['RMSE'] = self.train_rmse
		history['val_loss'] = self.test_lossses
		history['val_RMSE'] = self.test_RMSEs
		with open('History/' + SAVE_PATH + '.his', 'wb') as fs:
			pickle.dump(history, fs)
		log('History Saved: %s' % SAVE_PATH)

if __name__ == '__main__':
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	maker = ScipyMatMaker()
	trainMat = maker.ReadMat(TRAIN_FILE).toarray()
	testMat = maker.ReadMat(TEST_FILE).toarray()
	cvMat = maker.ReadMat(CV_FILE).toarray()
	with tf.Session(config=config) as sess:
		AttentiveAE = AttentiveAE(sess, trainMat, cvMat, testMat)
		AttentiveAE.run()
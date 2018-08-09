import numpy as np
from AutoRec import AutoEncoder
import ToolScripts.TimeLogger as logger
import ToolScripts.Emailer as mailer
import MakeData
from Params import *
import heapq
import pickle
from scipy.sparse import csr_matrix

def FuseMats(preMat, colMat):
	tem = (preMat == 0) * (colMat)
	logger.log('PreMat Datas: %d, ColMat Adding Datas: %d' % (np.sum((preMat!=0)), np.sum((tem!=0))))
	return preMat - tem

def InputOffset(inputMat):
	return inputMat - 3.0 * (inputMat != 0)

def RunExperiment():
	logger.log('Start')
	net = AutoEncoder()
	if LOAD_MODEL != None:
		net.loadModel()
		logger.log('Model Loaded')

	logger.log('Load Matrices')
	maker = MakeData.ScipyMatMaker()
	trainMat = maker.ReadMat(TRAIN_FILE)
	cvMat = maker.ReadMat(CV_FILE)
	colMat = maker.ReadMat(COL_FILE)

	metrics = ['loss', 'RMSE', 'val_loss', 'val_RMSE']
	history = dict()
	for metric in metrics:
		history[metric] = list()
	logger.log('Train Start')
	# trainHistory = net.trainSimp(trainMat, trainMat, (trainMat, cvMat))
	trainHistory = net.train(maker.GenerateSymmetricBatch(trainMat),
		validation_data=maker.GenerateAsymmetricBatch(trainMat, cvMat))
	for metric in metrics:
		history[metric] += trainHistory.history[metric]
	logger.log('Train Complete')

	del cvMat
	logger.log('Load Test')
	testMat = maker.ReadMat(TEST_FILE)
	# res = net.evaluateSimp(trainMat, testMat)
	res = net.evaluate(maker.GenerateAsymmetricBatch(trainMat, testMat))

	logger.log('Test End')
	logger.log('Loss: %f, RMSE: %f' % (res[0], res[1]))

	with open('History/' + SAVE_PATH + '.his', 'wb') as fs:
		pickle.dump(trainHistory.history, fs)
	logger.log('End')


	print('-----------------------------------------')
	logger.log('Model Save Path: %s' % SAVE_PATH)
	logger.log('Load Model: ' + str(LOAD_MODEL))
	logger.log('Latent Dim: %d' % LATENT_DIM)
	logger.log('Regularize Weight: %f' % REG_WEIGHT)
	logger.log('Batch Size: %d' % BATCH_SIZE)
	logger.log('COL Weight: %f' % COL_WEIGHT)
	mailer.SendMail(logger.logmsg, 'An Experiment Ends')
	# import ResultAnalyzer

if __name__ == '__main__':
	try:
		logger.saveDefault = True
		RunExperiment()
	except Exception as e:
		# mailer.SendMail(logger.logmsg + '\n' + str(e), 'An Error Has Occured in Your Experiment')
		raise e
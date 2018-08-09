import ToolScripts.DataProcessor as proc
import ToolScripts.TimeLogger as logger
import os
import numpy as np
import ToolScripts.Emailer as mailer
import gc
import pickle
from Params import *
import math
from scipy.sparse import csr_matrix
import scipy.sparse as sp

class DataDivider:
	def DivideData(self):
		temFile = 'ml-1m/shuffledRatings.csv'
		proc.RandomShuffle('ml-1m/ratings.dat', temFile, True)
		logger.log('End Shuffle')
		for rate in RATES:
			out1 = 'ml-1m/ratings_' + str(rate) + '_train.csv'
			out2 = 'ml-1m/ratings_' + str(rate) + '_test.csv'
			out3 = 'ml-1m/ratings_' + str(rate) + '_cv.csv'
			proc.SubDataSet(temFile, out1, out2, rate)
			proc.SubDataSet(out1, out1, out3, 0.9 )
		os.remove(temFile)

class ScipyMatMaker:
	def MakeOneMat(self, infile, outfile):
		data = list()
		rows = list()
		cols = list()
		with open(infile, 'r') as fs:
			for line in fs:
				arr = line.strip().split(DIVIDER)
				movieId = int(arr[1]) - 1
				userId = int(arr[0]) - 1
				rating = float(arr[2])
				rows.append(movieId)
				cols.append(userId)
				data.append(rating)
		mat = csr_matrix((data, (rows, cols)), shape=(MOVIE_NUM, USER_NUM))
		with open(outfile, 'wb') as fs:
			pickle.dump(mat, fs)

	def ReadMat(self, file):
		with open(file, 'rb') as fs:
			ret = pickle.load(fs)
		return ret

	def SliceMat(self, mat, cur):
		if cur + BATCH_SIZE <= MOVIE_NUM:
			return mat[cur: cur + BATCH_SIZE].toarray()
		return sp.vstack((mat[cur: MOVIE_NUM], mat[0: MOVIE_NUM - cur + 1])).toarray()

	def GenerateAsymmetricBatch(self, inputMat, labelMat):
		curid = 0
		while True:
			yield (self.SliceMat(inputMat, curid), self.SliceMat(labelMat, curid))
			curid = (curid + BATCH_SIZE) % MOVIE_NUM

	def GenerateSymmetricBatch(self, mat):
		curid = 0
		while True:
			tem = self.SliceMat(mat, curid)
			yield (tem, tem)
			curid = (curid + BATCH_SIZE) % MOVIE_NUM

	def MakeMats(self):
		trainfile = dataset + '/ratings_' + str(RATE) + '_train.csv'
		testfile = dataset + '/ratings_' + str(RATE) + '_test.csv'
		cvfile = dataset + '/ratings_' + str(RATE) + '_cv.csv'
		self.MakeOneMat(trainfile, TRAIN_FILE)
		self.MakeOneMat(testfile, TEST_FILE)
		self.MakeOneMat(cvfile, CV_FILE)

if __name__ == "__main__":
	logger.log('Start')
	maker = ScipyMatMaker()
	maker.MakeMats()
	logger.log('Sparse Matrix Made')
	# mailer.SendMail('No Error')
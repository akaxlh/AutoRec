from Params import *
import heapq
from MakeData import ScipyMatMaker
import pickle
from ToolScripts.TimeLogger import log
import numpy as np
from scipy.sparse import csr_matrix
import heapq
from scipy.sparse import csr_matrix

def Print(vec, file):
	with open(file, 'w') as fs:
		for val in vec:
			fs.write(str(val[0]) + ': ' + str(val[1]) + '\n')

def FastItemCF(ratingMat, coldMovies):
	neighbors = [list() for row in range(MOVIE_NUM)]
	userRates = [None] * USER_NUM
	for userId in range(USER_NUM):
		userRates[userId] = ratingMat[:, userId]
	for i in range(MOVIE_NUM):
		if not coldMovies[i]:
			continue
		temvec = (ratingMat[i].toarray())[0]
		relatedUsers = np.argwhere(temvec != 0)
		numerator = np.zeros((MOVIE_NUM, 1))
		denominator1 = np.zeros((MOVIE_NUM, 1))
		denominator2 = np.zeros((MOVIE_NUM, 1))
		cnt = np.zeros((MOVIE_NUM, 1))
		for user in relatedUsers:
			userId = user[0]
			temRate = userRates[userId].toarray()
			mask = (temRate != 0)
			numerator += temRate * temvec[userId]
			denominator1 += np.square(temRate)
			denominator2 += mask * np.square(temvec[userId])
			cnt += mask
		scores = np.ravel(numerator / (np.multiply(np.sqrt(denominator1), np.sqrt(denominator2)) + 1 - (numerator!=0)))
		neighbors[i] = [(row, scores[row], cnt[row, 0]) for row in range(MOVIE_NUM)]
		neighbors[i] = list(filter(lambda x: x[1]>COSINE_THRESHOLD, neighbors[i]))
		neighbors[i].sort(key=lambda x:x[1], reverse=True)
		if len(neighbors[i]) > KNN_K:
			neighbors[i] = neighbors[i][:KNN_K]
	with open('ml-1m/neighbors', 'wb') as fs:
		pickle.dump(neighbors, fs)
	return neighbors

def BinColMat(neighbors, ratingMat):
	data = list()
	rows = list()
	cols = list()
	rates = [None] * MOVIE_NUM
	for i in range(MOVIE_NUM):
		rates[i] = ratingMat[i]
	for i in range(MOVIE_NUM):
		scores = np.zeros((1, USER_NUM))
		cnt = np.zeros((1, USER_NUM))
		for neighbor in neighbors[i]:
			temvec = rates[neighbor[0]].toarray()
			scores += temvec
			cnt += (temvec != 0)
		scores = scores / (cnt + 1.0 - (cnt != 0))
		locs = np.argwhere(scores != 0.0)
		print(scores)
		for loc in locs:
			data.append(-scores[0][loc[1]])
			rows.append(i)
			cols.append(loc[1])
		# for neighbor in neighbors[i]:
		# 	temvec = rates[neighbor[0]].toarray()
		# 	posRates = (temvec == 5) + (temvec == 4) * 0.5
		# 	negRates = (temvec == 1) + (temvec == 2) * 0.5
		# 	pos += posRates
		# 	neg += negRates
		# diffLocs = (np.abs(pos - neg) > (pos + neg) * 0.8) * (pos + neg > RATING_THRESHOLD)
		# posLocs = np.argwhere((pos - neg) * diffLocs > 0)
		# negLocs = np.argwhere((neg - pos) * diffLocs > 0)
		# for loc in posLocs:
		# 	data.append(-1)
		# 	rows.append(i)
		# 	cols.append(loc[1])
		# for loc in negLocs:
		# 	data.append(-2)
		# 	rows.append(i)
		# 	cols.append(loc[1])
	log('Data Num: %d' % len(data))
	return csr_matrix((data, (rows, cols)), shape=(MOVIE_NUM, USER_NUM))

def Test(pred, label):
	err = 0.0
	n = 0.0
	unk = 0
	for movieId in range(MOVIE_NUM):
		vec = (label[movieId].toarray())[0]
		locs = np.argwhere(vec != 0)
		for loc in locs:
			userId = loc[0]
			trueScore = vec[userId]
			temPred = pred[movieId, userId]
			if temPred == 0.0:
				unk += 1
				continue
			err += np.square(temPred + trueScore)
			n += 1.0
	print('RMSE: ', np.sqrt(err/n))
	print('n: ', n)
	print('unknown: ', unk)
	print('err: ', err)

def GetColdMovies(ratingMat):
	rates = np.sum((ratingMat.toarray())!=0, 1)
	coldMovies = rates <= RATING_THRESHOLD
	return coldMovies

if __name__ == "__main__":
	log('Start')
	maker = ScipyMatMaker()
	ratingMat = maker.ReadMat(TRAIN_FILE)
	cvMat = maker.ReadMat(CV_FILE)
	log('Rating Mat Read')

	coldMovies = GetColdMovies(ratingMat)
	log('Cold Movies Got')

	neighbors = maker.ReadMat('ml-1m/neighbors')
	# neighbors = FastItemCF(ratingMat, coldMovies)
	log('Fast KNN Complete')
	# colMat = BinColMat(neighbors, ratingMat)
	colMat = maker.ReadMat(COL_FILE)
	print(colMat)
	log('Collaborattion Matrix Got')

	Test(colMat, cvMat)
	with open(COL_FILE, 'wb') as fs:
		pickle.dump(colMat, fs)

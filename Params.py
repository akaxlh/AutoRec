# Data Parameters
dataset = 'ml-1m'
RATE = 0.9
if dataset == 'ml-20m':
	USER_NUM = 138493
	MOVIE_NUM = 131262
	DIVIDER = ','
elif dataset == 'ml-1m':
	USER_NUM = 6040
	MOVIE_NUM = 3952
	DIVIDER = '::'

# Storage Parameters
LOAD_MODEL = None# 'AutoRec方法复现/Models/复现成功_L500_W.05_R.001_D.001_LAST'
# TRAIN_FILE = dataset + '/sparseMat/sparseMat_0.9_train.csv'
# TEST_FILE = dataset + '/sparseMat/sparseMat_0.9_test.csv'
# CV_FILE = dataset + '/sparseMat/sparseMat_0.9_cv.csv'
# TRANS_FILE = dataset + '/sparseMat/transSparseMat_0.9_train.csv'
# COL_FILE = dataset + '/sparseMat/sparseMat_0.9_col.csv'

TRAIN_FILE = dataset + '/mats/sparseMat_0.9_train.csv'
TEST_FILE = dataset + '/mats/sparseMat_0.9_test.csv'
CV_FILE = dataset + '/mats/sparseMat_0.9_cv.csv'
COL_FILE = dataset + '/mats/sparseMat_0.9_col.csv'

# Model Parameters
LATENT_DIM = 500
D_HIDDEN = 500
REG_WEIGHT = 0.05
LR = 0.001
DECAY = 0.001
BATCH_SIZE = 64
EPOCH = 200
KNN_K = 400
COSINE_THRESHOLD = 0.95
RATING_THRESHOLD = 10
COL_WEIGHT = 0.00
BIG_BATCH = 2048


hashval = 0
for param in dir():
	if not param.startswith('__'):
		val = hash(locals()[param])
		hashval = (hashval * 233 + val) % 100000000007
ModelName = 'impossible'

# SAVE_PATH = 'default.model'
SAVE_PATH = str(hashval) + ModelName + '.model'
import pickle
import ToolScripts.Plotter as plotter
import matplotlib.pyplot as plt
import numpy as np
from Params import *

colors = ['red', 'cyan', 'blue', 'green', 'black', 'magenta', 'yellow', 'pink', 'purple', 'chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon', 'gold', 'darkred']
lines = ['-', '--', '-.', ':']

sets = ['37136206697.model', '94193535568.model', '77822946976Baseline2000.model', '39125763032BaselineUsingGenerator.model', '32880317168tf_reg_0.1_batch_4.model', '36716700567tf_reg0.1_batch64_meanloss.model']
names = ['Baseline', 'Offseted', 'Baseline 2000', 'BaselineUsingGenerator 2000', 'tf_reg0.1_batch4', 'tf_reg0.1_batch64_meanloss']

for j in range(len(sets)):
	val = sets[j]
	name = names[j]
	print('val', val)
	with open('History/%s.his' % val, 'rb') as fs:
		res = pickle.load(fs)

	length = EPOCH
	temy = np.zeros((4, length))
	temy[0] = np.array(res['loss'][:length])
	temy[1] = np.array(res['RMSE'][:length])
	temy[2] = np.array(res['val_loss'][:length])
	temy[3] = np.array(res['val_RMSE'][:length])
	y = [[], [], [], []]
	smooth = 1
	for i in range(int(length/smooth)):
		if i*smooth+smooth-1 >= len(temy[0]):
			break
		for k in range(4):
			temsum = 0.0
			for l in range(smooth):
				temsum += temy[k][i*smooth+l]
			y[k].append(temsum / smooth)
	y = np.array(y)
	length = y.shape[1]
	x = np.zeros((4, length))
	for i in range(4):
		x[i] = np.array(list(range(length)))
	plt.figure(1)
	plt.subplot(221)
	plt.title('LOSS FOR TRAIN')
	plt.plot(x[0], y[0], color=colors[j], label=name)
	plt.legend()
	plt.subplot(222)
	plt.title('LOSS FOR VAL')
	plt.plot(x[2], y[2], color=colors[j], label=name)
	plt.legend()
	plt.subplot(223)
	plt.title('RMSE FOR TRAIN')
	plt.plot(x[1], y[1], color=colors[j], label=name)
	plt.legend()
	plt.subplot(224)
	plt.title('RMSE FOR VAL')
	plt.plot(x[3], y[3], color=colors[j], label=name)
	plt.legend()

plt.show()

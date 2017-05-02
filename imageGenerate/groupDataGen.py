
import numpy as np
import os
import random
from shutil import copyfile

# numGroup = 250
numGroup = 13
# numNongroup = 150
groupSym = {'!', '=', 'cos', 'div', 'geq', 'i', 'j', 'k', 'ldots', 'leq', \
			'lim', 'log', 'pi', 'pm', 'rightarrow', 'sin', 'sum', 'tan', \
			'theta', 'upper_E', 'upper_F', 'upper_I', 'upper_P', 'upper_R', \
			'upper_T', 'upper_Y', '5', 'x'}

# GENERATE DATA FOR GROUP CNN
def groupData(saveFolder, readTrainFolder):
	if not os.path.exists(saveFolder):
		os.mkdir(saveFolder)
	symbols = [sym for sym in os.listdir(readTrainFolder) \
				if not sym.startswith('.') and sym in groupSym]
	for sym in symbols:
		readSymPath = os.path.join(readTrainFolder, sym)
		imgNames = [img for img in os.listdir(readSymPath) if img.endswith('.png')]
		num_img = numGroup
		if len(imgNames) >= num_img:
			imgNames = random.sample(imgNames,num_img)
		else:
			cp_size = num_img / len(imgNames)
			reminding = num_img - cp_size * len(imgNames)
			padImgs = random.sample(imgNames,reminding)
			imgNames = np.repeat(imgNames, cp_size)
			imgNames = np.append(imgNames, padImgs)
		for i in xrange(len(imgNames)):
			imgReadPath = os.path.join(readSymPath, imgNames[i])
			cpPath = os.path.join(saveFolder, sym+'_'+str(i)+'.png')
			copyfile(imgReadPath, cpPath)

saveFolder = '../TestIMAGE_GROUP_TRUE'
readTrainFolder = '../TestIMAGE'
groupData(saveFolder, readTrainFolder)

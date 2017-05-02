##### HAND WRITTEN MATH EXPRESSION RECGONIZER #####

import os
import subprocess
import numpy as np
from skimage.filters import threshold_mean

# from cnn import cnn_predict
from vertical import vertical_merge
from horizontal import horizontal
from helper import get_img_names, group_img_npy, mergeGroups, sym_img_npy, symbols

folder = '../TestWhole'
saveVertiGP = 'verticGroup'
saveHorizGP = 'horizGroup'
saveSymbol = 'symGroup'
verticPredict = 'verticPredict'
horizPredict = 'horizPredict'
symPredict = 'symPredict'
groupModel = 'cnnGroupModel/model.ckpt'
classModel = 'cnnModel/model.ckpt'

imgNames = get_img_names(folder)
# imgNames = ['103_em_12.png']
# imgNames = ['65_alfonso.png']
writeFile = open('predictSymbols.txt', 'wr+')
for imgName in imgNames:
	print("Start Image: " + imgName)
	# VERTICAL GROUPING
	label, CCs, groups = vertical_merge(folder, imgName)
	if len(groups) > 0:
		# SAVE VERTICAL GROUPS
		savePath = os.path.join(saveVertiGP, imgName).replace('png', 'npy')
		group_img_npy(label, CCs, groups, savePath)
		# PREDICT ON VERTICAL GROUPING
		predictPath = os.path.join(verticPredict, imgName).replace('png', 'npy')
		# print predictPath
		process = subprocess.Popen(['python', 'cnnGroup.py', savePath, groupModel, predictPath])
		process.wait()
		predictions = np.load(predictPath)
		# print predictions
		# MERGE GROUPS BASED ON PREDICTIONS
		label = mergeGroups(label, groups, predictions)
	# HORIZONTAL GROUPING
	label, CCs, groups = horizontal(label)
	if len(groups) >0:
		# SAVE HORIZONTAL GROUPS
		savePath = os.path.join(saveHorizGP, imgName).replace('png', 'npy')
		group_img_npy(label, CCs, groups, savePath)
		# PREDICT ON HORIZONTAL GROUPING
		predictPath = os.path.join(horizPredict, imgName).replace('png', 'npy')
		process = subprocess.Popen(['python', 'cnnGroup.py', savePath, groupModel, predictPath])
		process.wait()
		predictions = np.load(predictPath)
		# print predictions
		# MERGE GROUPS BASED ON PREDICTIONS
		label = mergeGroups(label, groups, predictions)
	# SYMBOLS
	savePath = os.path.join(saveSymbol, imgName).replace('png', 'npy')
	sym_img_npy(label, savePath)
	# PREDICT FOR SYMBOLS
	predictPath = os.path.join(symPredict, imgName).replace('png', 'npy')
	process = subprocess.Popen(['python', 'cnnSymbol.py', savePath, classModel, predictPath])
	process.wait()
	# OUTPUT FINAL SYMBOLS
	predictions = np.load(predictPath)
	symbols(predictions, writeFile, imgName)
	print("Completed")


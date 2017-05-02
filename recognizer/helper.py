# HELPER FUNCTION FOR PUTTING EVERYTHING TOGETHER
# 
import os
import sys
import numpy as np
from scipy.misc import imresize
from skimage.filters import threshold_mean
from skimage import io, measure

NUM_CHANNELS = 1
IMAGE_SIZE = 32
PAD_RATIO = 0.1

def get_img_names(folder):
	return [img for img in os.listdir(folder) if img.endswith('.png')]


def group_img_npy(label, CCs, groups, savePath):
	num_images = len(groups)
	# print num_images
	data = np.ndarray(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),\
					  dtype=np.float32)
	idx = 0
	for group in groups:
		gpImg = np.zeros(label.shape)
		bound = [sys.maxint, sys.maxint, -1, -1]
		for i in xrange(len(group)):
			cc_idx = group[i]
			cc_bound = CCs[cc_idx].bbox
			bound[0] = min(bound[0], cc_bound[0])
			bound[1] = min(bound[1], cc_bound[1])
			bound[2] = max(bound[2], cc_bound[2])
			bound[3] = max(bound[3], cc_bound[3])
			gpImg += (label == (cc_idx+1))
		w_pad = int((bound[3] - bound[1]) * PAD_RATIO)
		h_pad = int((bound[2] - bound[0]) * PAD_RATIO)
		gpImg = gpImg[bound[0]-h_pad:bound[2]+h_pad, bound[1]-w_pad:bound[3]+w_pad]
		gpImg = (gpImg == 0)
		img_size = gpImg.shape
		padding = max(img_size) - min(img_size)
		pad_s = padding / 2
		pad_l = padding - pad_s
		if img_size[0] == max(img_size):
			gpImg = np.lib.pad(gpImg, ((0,0),(pad_s,pad_l)), \
							 'constant', constant_values=((1,1),(1,1)))
		else:
			gpImg = np.lib.pad(gpImg, ((pad_s,pad_l),(0,0)), \
							 'constant', constant_values=((1,1),(1,1)))
		gpImg = imresize(gpImg, (IMAGE_SIZE, IMAGE_SIZE))
		thresh = threshold_mean(gpImg)
		gpImg = (gpImg > thresh) * 255
		data[idx,:,:,0] = gpImg
		idx += 1
		np.save(savePath, data)
	return


def mergeGroups(labels, groups, predictions):
	trueThred = 0.5
	# compThred = 0.1
	final_groups = []
	predictions = predictions[:,1]
	groups = [groups[i] for i, pre in enumerate(predictions) if pre > trueThred]
	predictions = [pre for pre in predictions if pre > trueThred]
	# print predictions
	# deal with triple case
	# tripGpIdx = [i for i, group in enumerate(groups) if len(group) == 3]
	# for gpIdx in tripGpIdx:
	# 	curGp = groups[gpIdx]
	# 	del groups[gpIdx]
	# 	conflictGpIdx = [i for i, gp in enumerate(groups) if len(gp) == 2 \
	# 					 and (gp[0] in curGp or gp[1] in curGp)]
	# 	if len(np.argwhere(predictions[conflictGpIdx] > compThred + predictions[gpIdx])) == 0:
	# 		final_groups.append(curGp)
	# 		groups = [gp for i, gp in enumerate(groups) if i not in conflictGpIdx]
	# gredily pick
	sortIdx = np.argsort(predictions)[::-1]
	groups = [groups[i] for i in sortIdx]
	# print groups
	# print sortIdx
	deleteIdx = []
	for i in xrange(len(groups)-1):
		if i not in deleteIdx:
			conflictGpIdx = [i for i, gp in enumerate(groups[(i+1):]) \
							 if any(ele in gp for ele in groups[i])]
			deleteIdx.extend(conflictGpIdx)
	final_groups = [groups[i] for i in xrange(len(groups)) if i not in deleteIdx]
	# merge final groups
	for group in final_groups:
		sortedIdx = sorted(group)
		mergeIdx = sortedIdx[0]
		for idx in sortedIdx[1:]:
			coors = np.argwhere(labels == (idx+1))
			labels[coors[:,0],coors[:,1]] = mergeIdx+1
	# modify labels
	uni_label = np.unique(labels)
	num_label = len(uni_label) - 1
	for i in xrange(1,num_label+1):
		if uni_label[i] != i:
			sIdx = np.argwhere(labels==uni_label[i])
			labels[sIdx[:,0],sIdx[:,1]] = i

	return labels


def sym_img_npy(label, savePath):
	CCs = measure.regionprops(label)
	num_sym = len(CCs)
	# print num_sym
	data = np.ndarray(shape=(num_sym, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),\
					  dtype=np.float32)
	idx = 0
	for i in xrange(1,len(CCs)+1):
		symImg = np.zeros(label.shape)
		symImg += (label == i)
		bound = CCs[i-1].bbox
		w_pad = int((bound[3] - bound[1]) * PAD_RATIO)
		h_pad = int((bound[2] - bound[0]) * PAD_RATIO)
		img_size = label.shape
		x_min = bound[0]-h_pad
		x_max = bound[2]+h_pad
		y_min = bound[1]-w_pad
		y_max = bound[3]+w_pad
		if x_min < 0: x_min = 0
		if x_max > img_size[0]: x_max = img_size[0]
		if y_min < 0: y_min = 0
		if y_max > img_size[1]: y_max = img_size[1]
		symImg = symImg[x_min:x_max, y_min:y_max]
		symImg = (symImg == 0)
		img_size = symImg.shape
		padding = max(img_size) - min(img_size)
		pad_s = padding / 2
		pad_l = padding - pad_s
		if img_size[0] == max(img_size):
			symImg = np.lib.pad(symImg, ((0,0),(pad_s,pad_l)), \
							 'constant', constant_values=((1,1),(1,1)))
		else:
			symImg = np.lib.pad(symImg, ((pad_s,pad_l),(0,0)), \
							 'constant', constant_values=((1,1),(1,1)))
		symImg = imresize(symImg, (IMAGE_SIZE, IMAGE_SIZE))
		thresh = threshold_mean(symImg)
		symImg = (symImg > thresh) * 255
		data[idx,:,:,0] = symImg
		idx += 1
		np.save(savePath, data)
	return


def symbols(predictions, writeFile, imgName):
	label_list = np.array(['-', ',', '!', '(', ')', '[', ']', '{', '}', '+', '=', '|', '0', \
				 '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'alpha', 'b', 'beta', 'c', \
				 'cos', 'd', 'delta', 'div', 'e', 'f', 'fslash', 'g', 'gamma', 'geq', 'gt', 'h', \
				 'i', 'infty', 'int', 'j', 'k', 'l', 'ldots', 'leq', 'lim', 'log', 'lt', 'm', \
				 'mu', 'n', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'r', 'rightarrow', \
				 's', 'sigma', 'sin', 'sqrt', 'sum', 't', 'tan', 'theta', 'times', 'u', 'upper_A', \
				 'upper_B', 'upper_C', 'upper_E', 'upper_F', 'upper_G', 'upper_H', 'upper_I',\
				 'upper_L', 'upper_M', 'upper_N', 'upper_P', 'upper_R', 'upper_S', 'upper_T', \
				 'upper_V', 'upper_X', 'upper_Y', 'v', 'w', 'x', 'y', 'z'])
	symIdx = np.argmax(predictions, axis=1)
	predLabels = label_list[symIdx]
	writeFile.write(imgName + ':' + ' '.join(predLabels) + '\n')


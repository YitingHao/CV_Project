# GENERATE GROUP IMAGES

import os
import sys
import vertical
import horizontal
import numpy as np
from skimage import io
from skimage.filters import threshold_mean
from scipy.misc import imresize
import matplotlib.pyplot as plt

targetFolder = '../test'
readFolder = '../TrainWhole'

pad_ratio = 0.1
norm_size = 32

def drawGroup(label, CCs, groups, savePath):
	img_idx = 0
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
		w_pad = int((bound[3] - bound[1]) * pad_ratio)
		h_pad = int((bound[2] - bound[0]) * pad_ratio)
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
		gpImg = imresize(gpImg, (norm_size, norm_size))
		thresh = threshold_mean(gpImg)
		gpImg = (gpImg > thresh) * 255
		splitsPath =  savePath.split('.')
		del splitsPath[-1]
		imgPath = '.'.join(splitsPath) + '_' + str(img_idx) + '.png'
		img_idx += 1
		io.imsave(imgPath, gpImg)
	return

	# idxs = []
	# idx = 0
	# lineWidth = 2
	# for group in groups:
	# 	gpImg = np.zeros(label.shape)
	# 	bound = [sys.maxint, sys.maxint, -1, -1]
	# 	for i in xrange(len(group)):
	# 		cc_idx = group[i]
	# 		cc_bound = CCs[cc_idx].bbox
	# 		bound[0] = min(bound[0], cc_bound[0])
	# 		bound[1] = min(bound[1], cc_bound[1])
	# 		bound[2] = max(bound[2], cc_bound[2])
	# 		bound[3] = max(bound[3], cc_bound[3])
	# 		gpImg += (label == (cc_idx+1))
	# 	idxs = np.argwhere(gpImg == 1)
	# 	fig = plt.figure()
	# 	plt.gca().set_aspect('equal', adjustable='box')
	# 	plt.axis('off')
	# 	w_pad = (bound[2] - bound[0]) * pad_ratio
	# 	h_pad = (bound[3] - bound[1]) * pad_ratio
	# 	plt.xlim(bound[0]-w_pad,bound[2]+w_pad)
	# 	plt.ylim(-bound[3]-h_pad,-bound[1]+h_pad)
	# 	x = idxs[:,0]
	# 	y = idxs[:,1] * -1
	# 	plt.plot(x, y, 'k', linewidth=lineWidth)
	# 	plt.show()
	# 	wholeImg = savePath
	# 	fig.savefig(wholeImg, bbox_inches='tight', pad_inches=0)
	# 	plt.close(fig)
	# 	idx += 1


# imgNames = [img for img in os.listdir(readFolder) if img.endswith('.png')]
imgNames = ['101_mijail.png']

#### VERTICAL MERGE ####
for imgName in imgNames:
	savePath = os.path.join(targetFolder, imgName)
	label, CCs, groups = vertical.vertical_merge(readFolder, imgName)
	drawGroup(label, CCs, groups, savePath)

#### HORIZONTAL MERGE ####
# for imgName in imgNames:
# 	savePath = os.path.join(targetFolder, imgName)
# 	label, CCs, groups = horizontal.horizontal(readFolder, imgName)
# 	drawGroup(label, CCs, groups, savePath)

# FUNCIONTS DEFINED FOR INTERPRETING Inkml FORMAT DATA INTO IMAGES

import re
import numpy as np
import sys
import os
from skimage import io
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.color import rgb2gray
from scipy.misc import imresize
from skimage.filters import threshold_mean

# BUILD UP TUPLE LIST: (GROUP(SYMBOL), TRACES INDEX)
def groupIdx(groupIdx_tup, line, sym_regex, idx_regex):
	sym_res = sym_regex.match(line)
	if sym_res:
		# add a new group as tuple
		symbol = sym_res.group(1)
		groupIdx_tup.append((symbol,[]))
		return
	idx_res = idx_regex.match(line)
	if idx_res:
		# add index into the last group
		idx = int(idx_res.group(1))
		groupIdx_tup[-1][1].append(idx)
		return

# BUILD UP A MAP: {TRACE INDEX: MATRIX(N,2)}
# RETURN WHETHER ENTER GROUPING ZONE OR NOT
def traceIdx(vec_map, line, cor_regex, gp_regex, f, mode):
	cor_res = cor_regex.match(line)
	if cor_res:
		idx = int(cor_res.group(1))
		if mode == 'MfrDB':
			vector = cor_res.group(2).replace(',','').split()
			vector = np.array([float(x) for i,x in enumerate(vector) if (i+1)%3 != 0])
		elif mode == 'KAIST':
			vector = cor_res.group(2).replace(',','').split()
			vector = np.array([float(x) for x in vector])
		else:
			vector = next(f).replace(',','').split()
			vector = np.array([float(x) for x in vector])
		vec_map[idx] = vector
		return False
	if gp_regex.match(line):
		return True
	return False

# GET IMAGE BOUNDING
# RETURN [Min_X, Min_Y, Max_X, Max_Y]
def bounding(groupIdx_tup, vec_map):
	bounds = []
	for (symbol, idxs) in groupIdx_tup:
		bound = [sys.maxint, sys.maxint, -1, -1]
		for idx in idxs:
			vec = vec_map[idx]
			x = vec[::2]
			y = vec[1::2]
			bound[0] = min(bound[0], min(x))
			bound[1] = min(bound[1], min(y))
			bound[2] = max(bound[2], max(x))
			bound[3] = max(bound[3], max(y))
		bounds.append(bound)
	bounds = np.array(bounds)
	imgBound = [min(bounds[:,0]), min(bounds[:,1]), max(bounds[:,2]), max(bounds[:,3])]
	bounds = np.vstack((bounds, imgBound))
	return bounds

# GENERATE WHOLE IMAGES
def wholeImgGen(groupIdx_tup, vec_map, bounds, \
				lineWidth, pad_ratio, imgName, saveDir, wholeImgFolder):
	fig = plt.figure()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.axis('off')
	w_pad = (bounds[-1,2] - bounds[-1,0]) * pad_ratio
	h_pad = (bounds[-1,3] - bounds[-1,1]) * pad_ratio
	plt.xlim(bounds[-1,0]-w_pad,bounds[-1,2]+w_pad)
	plt.ylim(-bounds[-1,3]-h_pad,-bounds[-1,1]+h_pad)
	for (symbol, idxs) in groupIdx_tup:
		for idx in idxs:
			vec = vec_map[idx]
			x = vec[::2]
			y = vec[1::2] * -1
			plt.plot(x, y, 'k', linewidth=lineWidth)
	# plt.show()
	wholeImg = saveDir + '/' + wholeImgFolder + '/' + imgName + '.png'
	fig.savefig(wholeImg, bbox_inches='tight', pad_inches=0)
	plt.close(fig)
	return

# GENERATE PART IMAGES
def partImgGen(groupIdx_tup, vec_map, bounds, lineWidth, pad_ratio, imgName, saveDir):
	for i, tp in enumerate(groupIdx_tup):
		fig = plt.figure()
		w_pad = (bounds[i,2] - bounds[i,0]) * pad_ratio
		h_pad = (bounds[i,3] - bounds[i,1]) * pad_ratio
		plt.gca().set_aspect('equal', adjustable='box')
		plt.xlim(bounds[i,0]-w_pad,bounds[i,2]+w_pad)
		plt.ylim(-bounds[i,3]-h_pad,-bounds[i,1]+h_pad)
		symbol = tp[0].split('\\')[-1]
		if symbol == '/':
			symbol = 'fslash'
		if symbol == '':
			symbol = 'bslash'
		for idx in tp[1]:
			vec = vec_map[idx]
			x = vec[::2]
			y = vec[1::2] * -1
			plt.plot(x, y, 'k',linewidth=lineWidth)
		plt.axis('off')

		# canvas = plt.get_current_fig_manager().canvas
		# agg = canvas.switch_backends(FigureCanvasAgg)
		# agg.draw()
		# s = agg.tostring_rgb()
		# l,b,w,h = agg.figure.bbox.bounds
		# w, h = int(w), int(h)
		# img = np.fromstring(s, dtype=np.uint8, sep='')
		# img = img.reshape([h,w,3])
		# norm_size = 32
		# img = rgb2gray(img)
		# img_size = img.shape
		# padding = max(img_size) - min(img_size)
		# pad_s = padding / 2
		# pad_l = padding - pad_s
		# if img_size[0] == max(img_size):
		# 	img = np.lib.pad(img, ((0,0),(pad_s,pad_l)), 'edge')
		# else:
		# 	img = np.lib.pad(img, ((pad_s,pad_l),(0,0)), 'edge')
		# img = imresize(img, (norm_size, norm_size))
		# thresh = threshold_mean(img)
		# img = (img > thresh) * 255

		symbolDir = saveDir + '/' + symbol
		if not os.path.exists(symbolDir):
			os.makedirs(symbolDir)
		subImgName = symbol + '_' + imgName + '_0'
		imgPath = os.path.join(symbolDir, subImgName+'.png')
		while os.path.exists(imgPath):
			symbol_idx = int(subImgName.split('_')[-1]) + 1
			subImgName = symbol + '_' + imgName + '_' + str(symbol_idx)
			imgPath = os.path.join(symbolDir, subImgName+'.png')
		# plt.show()
		fig.savefig(imgPath, bbox_inches='tight', pad_inches=0)
		# io.imsave(imgPath, img)
		plt.close(fig)
	return

# EXTRACT IMAGES FROM A INKML FILE
def extract_images(filePath, saveDir, wholeImgFolder, mode):
	# regex setting
	cor_pattern = '^<trace id="(.*)">$'
	group_pattern = '^<traceGroup xml:id="(.*)">$'
	symbol_pattern = '^<annotation type="truth">(.*)<\/annotation>$'
	groupIdx_pattern = '<traceView traceDataRef="(.*)"\/>'
	if mode != 'Normal':
		cor_pattern = '^<trace id="(.*)">(.*)</trace>$'
		groupIdx_pattern = '<traceView traceDataRef="(.*)" \/>'
	cor_regex = re.compile(cor_pattern)
	gp_regex = re.compile(group_pattern)
	sym_regex = re.compile(symbol_pattern)
	idx_regex = re.compile(groupIdx_pattern)
	# paras
	lineWidth = 4
	pad_ratio = 0.1
	start_gp = False
	groupIdx_tup = []
	vec_map = {}
	with open(filePath) as f:
		for line in f:
			line = line.strip()
			if start_gp:
				# generate tuple list: group - trace index
				groupIdx(groupIdx_tup, line, sym_regex, idx_regex)
			else:
				# generate map: trace index - matrix
				start_gp = traceIdx(vec_map, line, cor_regex, gp_regex, f, mode)
	del groupIdx_tup[0]
	# bounds
	bounds = bounding(groupIdx_tup, vec_map)
	imgName = filePath.split('/')[-1].split('.')[0]
	wholeImgGen(groupIdx_tup, vec_map, bounds, lineWidth, pad_ratio, \
		   imgName, saveDir, wholeImgFolder)
	partImgGen(groupIdx_tup, vec_map, bounds, lineWidth, pad_ratio, imgName, saveDir)
	return


# fig.delaxes(ax2)
# extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

# EXTRACT DATA FOR GROUP CNN

import os
import numpy as np
from skimage import io

IMAGE_SIZE = 32
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_PER_LABEL = 14000
# NUM_PER_LABEL = 717

# EXTACT DATA AND LABELS
def extract_data_labels(folderTrue, folderFalse, outData, outLabel):
	# initialzie data and labels
	data = np.ndarray(shape=(NUM_PER_LABEL, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), \
						dtype=np.float32)
	labels = np.zeros(shape=(NUM_PER_LABEL,), dtype=np.int64)
	# generate data and labels
	imgIdx = 0
	trueImgs = [img for img in os.listdir(folderTrue) if img.endswith('.png')]
	falseImgs = [img for img in os.listdir(folderFalse) if img.endswith('.png')]
	for img in trueImgs:
		imgPath = os.path.join(folderTrue, img)
		imgData = io.imread(imgPath)
		data[imgIdx,:,:,0] = imgData
		labels[imgIdx] = 1
		imgIdx = imgIdx + 1
	for img in falseImgs:
		imgPath = os.path.join(folderFalse, img)
		imgData = io.imread(imgPath)
		data[imgIdx,:,:,0] = imgData
		labels[imgIdx] = 0
		imgIdx = imgIdx + 1
	# shuffle
	idx = np.arange(NUM_PER_LABEL)
	np.random.shuffle(idx)
	data = data[idx,:,:,:]
	labels = labels[idx]
	# save into npy files for future usages
	np.save(outData, data)
	np.save(outLabel, labels)
	print data.shape
	print labels.shape
	return data, labels


folderTrue = '../TrainIMAGE_GROUP_TRUE'
folderFalse = '../TrainIMAGE_GROUP_VERTIC_FAULT'

outData = 'train_data_vertic.npy'
outLabel = 'train_label_vertic.npy'

data, labels = extract_data_labels(folderTrue, folderFalse, outData, outLabel)
print('Complete')

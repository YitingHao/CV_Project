# EXTRACT DATA

import os
import numpy as np
from skimage import io

IMAGE_SIZE = 32
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
# NUM_LABELS = 10
# VALIDATION_SIZE = 5000  # Size of the validation set.
# SEED = 66478  # Set to None for random seed.
# BATCH_SIZE = 64
# NUM_EPOCHS = 10
# EVAL_BATCH_SIZE = 64
# EVAL_FREQUENCY = 100  # Number of steps between evaluations.

# GET TOTAL NUMBER OF IMAGES
def get_num_images(folder):
	num_images = 0
	symbols = [sym for sym in os.listdir(folder) if not sym.startswith('.')]
	for sym in symbols:
		symPath = os.path.join(folder, sym)
		imgs = [img for img in os.listdir(symPath) if img.endswith('.png')]
		num_images = num_images + len(imgs)
	return num_images

# EXTACT DATA AND LABELS
def extract_data_labels(folder, num_images, outData, outLabel):
	label_map = {'-':0, ',':1, '!':2, '(':3, ')':4, '[':5, ']':6, '{':7, '}':8,\
				 '+':9, '=':10, '|':11, '0':12, '1':13, '2':14, '3':15, '4':16,\
				 '5':17, '6':18, '7':19, '8':20, '9':21, 'a':22, 'alpha':23,\
				 'b':24, 'beta':25, 'c':26, 'cos':27, 'd':28, 'delta':29, 'div':30,\
				 'e':31, 'f':32, 'fslash':33, 'g':34, 'gamma':35, 'geq':36, 'gt':37,\
				 'h':38, 'i':39, 'infty':40, 'int':41, 'j':42, 'k':43, 'l':44,\
				 'ldots':45, 'leq':46, 'lim':47, 'log':48, 'lt':49, 'm':50, 'mu':51,\
				 'n':52, 'neq':53, 'o':54, 'p':55, 'phi':56, 'pi':57, 'pm':58,\
				 'prime':59, 'q':60, 'r':61, 'rightarrow':62, 's':63, 'sigma':64,\
				 'sin':65, 'sqrt':66, 'sum':67, 't':68, 'tan':69, 'theta':70,\
				 'times':71, 'u':72, 'upper_A':73, 'upper_B':74, 'upper_C':75,\
				 'upper_E':76, 'upper_F':77, 'upper_G':78, 'upper_H':79, 'upper_I':80,\
				 'upper_L':81, 'upper_M':82, 'upper_N':83, 'upper_P':84, 'upper_R':85,\
				 'upper_S':86, 'upper_T':87, 'upper_V':88, 'upper_X':89, 'upper_Y':90,\
				 'v':91, 'w':92, 'x':93, 'y':94, 'z':95}
	# initialzie data and labels
	data = np.ndarray(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), \
						dtype=np.float32)
	labels = np.zeros(shape=(num_images,), dtype=np.int64)
	# generate data and labels for each image in each symbol folder
	imgIdx = 0
	symbols = [sym for sym in os.listdir(folder) if not sym.startswith('.')]
	for sym in symbols:
		symPath = os.path.join(folder, sym)
		imgs = [img for img in os.listdir(symPath) if img.endswith('.png')]
		for img in imgs:
			imgPath = os.path.join(symPath, img)
			imgData = io.imread(imgPath)
			data[imgIdx,:,:,0] = imgData
			labels[imgIdx] = label_map[sym]
			imgIdx = imgIdx + 1
	# shuffle
	idx = np.arange(num_images)
	np.random.shuffle(idx)
	data = data[idx,:,:,:]
	labels = labels[idx]
	# save into npy files for future usages
	np.save(outData, data)
	np.save(outLabel, labels)
	return data, labels


train_folder = '../TrainIMAGE'
outData = 'train_data.npy'
outLabel = 'train_label.npy'
num_train_images = get_num_images(train_folder)
train_data, train_labels = extract_data_labels(train_folder, num_train_images, \
											   outData, outLabel)
print('debug')


import numpy as np
from skimage import io
from scipy.misc import imresize
from skimage.filters import threshold_mean
from skimage.color import rgb2gray

# RESIZE IMAGE
def resizeImg(filePath, savePath, norm_size):
	img = rgb2gray(io.imread(filePath))
	img_size = img.shape
	padding = max(img_size) - min(img_size)
	pad_s = padding / 2
	pad_l = padding - pad_s
	if img_size[0] == max(img_size):
		img = np.lib.pad(img, ((0,0),(pad_s,pad_l)), \
						'constant', constant_values=((1,1),(1,1)))
	else:
		img = np.lib.pad(img, ((pad_s,pad_l),(0,0)), \
						'constant', constant_values=((1,1),(1,1)))
	img = imresize(img, (norm_size, norm_size))
	thresh = threshold_mean(img)
	img = (img > thresh) * 255
	io.imsave(savePath, img)
	return

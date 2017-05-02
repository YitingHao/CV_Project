
import os
from skimage import io
import numpy as np

# CHECK NULL IMAGES IN SUB-FOLDERS
def checkRemoveNull(dir, recFile):
	f = open(recFile, 'wr+')
	symbols = [sym for sym in os.listdir(dir) if not sym.startswith('.')]
	for sym in symbols:
		f.write(sym + ':\n')
		symPath = os.path.join(dir, sym)
		imgs = [img for img in os.listdir(symPath) if img.endswith('.png')]
		for img in imgs:
			imgPath = os.path.join(symPath, img)
			imgData = io.imread(imgPath)
			if np.amin(imgData) == np.amax(imgData):
				f.write(imgPath + '\n')
				os.remove(imgPath)
	f.close()



directory = '../TestIMAGE_NORM'
recFile = 'emptyFiles_norm_test.txt'
checkRemoveNull(directory, recFile)

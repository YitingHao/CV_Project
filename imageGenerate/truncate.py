# TRUNCATE TRAINING DATA

import os
from shutil import rmtree
import random

def trunc(folder, minThresh, maxThresh):
	symbols = [sym for sym in os.listdir(folder) if not sym.startswith('.')]
	for sym in symbols:
		symPath = os.path.join(folder, sym)
		imgs = [img for img in os.listdir(symPath) if img.endswith('.png')]
		size = len(imgs)
		if size < minThresh:
			rmtree(symPath)
		elif size > maxThresh:
			random.shuffle(imgs)
			deleteImgs = imgs[maxThresh:]
			for img in deleteImgs:
				imgPath = os.path.join(symPath, img)
				os.remove(imgPath)
		print("Finish Symbol: " + sym)



folder = 'Merge'
minThresh = 30
maxThresh = 300
trunc(folder, minThresh, maxThresh)

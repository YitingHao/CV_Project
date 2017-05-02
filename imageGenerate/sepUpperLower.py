
import os
from shutil import copy

def extract_Upper(folder):
	charSymbols = [sym for sym in os.listdir(folder) \
						if len(sym)==1 and sym.isalpha()]
	for char in charSymbols:
		symbolFold = os.path.join(folder, char)
		upperChar = char.upper()
		# make upper symbol char directory
		upperSymFold = os.path.join(folder, 'upper_'+upperChar)
		if not os.path.exists(upperSymFold):
			os.mkdir(upperSymFold)
		# go through all images
		imgs = [img for img in os.listdir(symbolFold) if img.endswith('.png')]
		for img in imgs:
			if img.split('_')[0] == upperChar:
				imgPath = os.path.join(symbolFold, img)
				copy(imgPath,upperSymFold)
				os.remove(imgPath)
		print('Finish Symbol: ' + char)

folder = '../TestIMAGE_NORM'
extract_Upper(folder)

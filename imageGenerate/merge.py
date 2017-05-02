# MERGE IMAGES WITH SAME LABELS IN TWO FOLDERS (INTO FOLDER 1)
# Script parameters: two folders name

import os
from shutil import copy, copytree
from skimage import io
import numpy as np

# MERGING LIST OF FOLDERS
def mergeList(folder1, dir, folderList):
	for folder in folderList:
		folder2 = os.path.join(directory, folder)
		subFolder1 = set([f.lower() for f in os.listdir(folder1) if not f.startswith('.')])
		subFolder2 = [f.lower() for f in os.listdir(folder2) if not f.startswith('.')]
		for subfolder in subFolder2:
			readPath = os.path.join(folder2,subfolder)
			savePath = os.path.join(folder1,subfolder)
			if subfolder in subFolder1:
				# exist in folder1 already
				files = [f for f in os.listdir(readPath) if f.endswith('.png')]
				for f in files:
					filePath = os.path.join(readPath, f)
					copy(filePath,savePath)
			else:
				copytree(readPath, savePath)
		print('Finish Merge Folder: ' + folder)

# CHECKER NULL IMAGES
def checkNull(folderPath, recFile):
	f = open(recFile, 'wr+')
	symbols = [sym for sym in os.listdir(folderPath) if not sym.startswith('.')]
	for sym in symbols:
		f.write(sym + ':\n')
		symFolder = os.path.join(folderPath, sym)
		imgs = [img for img in os.listdir(symFolder) if img.endswith('.png')]
		for img in imgs:
			imgPath = os.path.join(symFolder, img)
			img = io.imread(imgPath)
			if np.amin(img) == np.amax(img):
				f.write(imgPath + '\n')
		print('Finish Checking Symbol: ' + sym)
	f.close()


# para
folder1 = 'TrainIMAGE_NORM/Merge'
directory = 'TrainIMAGE_NORM'
folderList = ['expressmatch', 'extension', 'HAMEX', 'KAIST', 'MathBrush', "MfrDB"]

# merge first
# mergeList(folder1, directory, folderList)

# check null images 
recFile = 'emptyFiles_merge.txt'
checkNull(folder1, recFile)

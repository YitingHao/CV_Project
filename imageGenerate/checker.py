
import os
import re
import numpy as np
from skimage import io
import interpretInkml
from resize import resizeImg

# TRAIN_DATA: CHECK WHETHER ALL FORMATS ARE RIGHT BASED ON GENERATED IMAGES
def formatChecker(dir, recFile):
	f = open(recFile, 'wr+')
	wholeImg = 'wholeImgs'
	folders = os.listdir(dir)
	for folder in folders:
		if not folder.startswith('.'):
			f.write(folder + '\n')
			wholeImgPath = os.path.join(dir,folder,wholeImg)
			imgs = os.listdir(wholeImgPath)
			for imgName in imgs:
				if imgName.endswith('.png'):
					imgPath = os.path.join(wholeImgPath, imgName)
					img = io.imread(imgPath)
					if np.amin(img) == np.amax(img):
						f.write(imgName + '\n')
		print("Finish folder: " + folder)
	f.close()

# CHECK NULL IMAGES IN ALL SUB-IMAGES
def normFormChecker(dir, recFile):
	f = open(recFile, 'wr+')
	folders = os.listdir(dir)
	for folder in folders:
		if not folder.startswith('.'):
			f.write(folder + ':\n')
			folderPath = os.path.join(dir, folder)
			subfolders = os.listdir(folderPath)
			for subfolder in subfolders:
				if not subfolder.startswith('.'):
					f.write('   ' + subfolder + '-\n')
					subfoldPath = os.path.join(folderPath, subfolder)
					imgs = os.listdir(subfoldPath)
					for imgName in imgs:
						imgPath = os.path.join(subfoldPath, imgName)
						img = io.imread(imgPath)
						if np.amin(img) == np.amax(img):
							f.write('      ' + imgName + '\n')
					print("Finish subfolder: " + subfolder)
			print("Finish folder: " + folder)
	f.close()

# DICTIONARY: EMPTY FILES FOR EACH FOLDER
def dicImages(txtFile):
	dic = {}
	cur_folder = ''
	folderPattern = ''
	with open(txtFile, 'r') as f:
		for line in f:
			line = line.strip()
			if line.endswith(':'):
				folderName = line.split(':')[0]
				dic[folderName] = set()
				cur_folder = folderName
			elif line.endswith('.png'):
				dic[cur_folder].add(line.split('.')[0])
	return dic

# DELETE SUB-IMAGES IN A DIRECTORY
def deleteSubImages(dir, dic):
	for key in dic:
		imgNames = dic[key]
		folderPath = os.path.join(dir, key)
		subFolders = os.listdir(folderPath)
		for subFolder in subFolders:
			if not subFolder.startswith('.') and subFolder != 'wholeImgs':
				subFolderPath = os.path.join(folderPath, subFolder)
				imgs = os.listdir(subFolderPath)
				for img in imgs:
					if img.endswith('.png'):
						imgList = img.split('_')
						del imgList[0]
						del imgList[-1]
						imgName = '_'.join(imgList)
						if imgName in imgNames:
							os.remove(os.path.join(subFolderPath,img))
							print('delete file: ' + imgName + ' in ' + subFolderPath)

# REMOVE NULL SUB-IMAGES
def removeEmptySub(dir, txtFile):
	folder = ''
	with open(txtFile, 'r') as f:
		for line in f:
			line = line.strip()
			if line.endswith(':'):
				folder = line.split(':')[0]
			elif line.endswith('.png'):
				subfolder = line.split('_')[0]
				imgPath = os.path.join(dir, folder, subfolder, line)
				os.remove(imgPath)
				print('delete file: ' + imgPath)


def removeEmpty(folder):
	imgs = [img for img in os.listdir(folder) if img.endswith('.png')]
	for imgName in imgs:
		imgPath = os.path.join(folder, imgName)
		img = io.imread(imgPath)
		if np.amin(img) == np.amax(img):
			os.remove(imgPath)


folder = '../VertiFault'
removeEmpty(folder)

# CHECK INCORRECT FORMAT
# directory = 'TrainIMAGE'
# recFile = 'emptyFiles.txt'
# formatChecker(directory, recFile)

# GET DICTIONARY - FOLDERNAME : SET OF IMAGE NAMES
# dic_images = dicImages(recFile)
# keys = []
# for key in dic_images:
# 	if len(dic_images[key]) == 0:
# 		keys.append(key)
# for key in keys:
# 	del dic_images[key]

# # DELETE INCORRECT SUB-IMAGES
# deleteSubImages(directory, dic_images)

# RE-GENERATE IMAGES USING "NORMAL" MODE
# mode = 'Normal'
# saveFolder = 'TrainIMAGE'
# wholeImgFolder = 'wholeImgs'
# readDir = 'TrainINKML'
# for key in dic_images:
# 	files = dic_images[key]
# 	readFoldPath = os.path.join(readDir, key)
# 	fileNames = os.listdir(readFoldPath)
# 	for f in fileNames:
# 		if f.split('.')[0] in files:
# 			filePath = os.path.join(readFoldPath, f)
# 			saveDir = os.path.join(saveFolder,key)
# 			print('Start Image: ' + filePath)
# 			interpretInkml.extract_images(filePath, saveDir, wholeImgFolder, mode)
# 			print('Complete!')

# CHECK AGAIN: THERE SHOULDN'T ANY MISTAKE
# recFile = 'emptyFiles_after.txt'
# formatChecker(directory, recFile)

# NORMALIZE NEW ADDED IMAGES
# readDir = 'TrainIMAGE'
# saveDir = 'TrainIMAGE_NORM'
# norm_size = 32
# for key in dic_images:
# 	imgNames = dic_images[key]
# 	readFoldPath = os.path.join(readDir, key)
# 	saveFoldPath = os.path.join(saveDir, key)
# 	subFolders = os.listdir(readFoldPath)
# 	for subFolder in subFolders:
# 		if not subFolder.startswith('.') and subFolder != 'wholeImgs':
# 			readSubFold = os.path.join(readFoldPath, subFolder)
# 			saveSubFold = os.path.join(saveFoldPath, subFolder)
# 			imgs = os.listdir(readSubFold)
# 			for img in imgs:
# 				if img.endswith('.png'):
# 					imgList = img.split('_')
# 					del imgList[0]
# 					del imgList[-1]
# 					imgName = '_'.join(imgList)
# 					if imgName in imgNames:
# 						readImg = os.path.join(readSubFold, img)
# 						savePath = os.path.join(saveSubFold, img)
# 						resizeImg(readImg, savePath, norm_size)

# CHECK INCORRECT FORMAT: THERE SHOULDN'T ANY MISTAKE
# directory = 'TrainIMAGE_NORM'
# recFile = 'emptyFiles_norm.txt'
# normFormChecker(directory, recFile)

# REMOVE INCORRECT FORMAT
# directory = 'TrainIMAGE_NORM'
# recFile = 'emptyFiles_norm.txt'
# removeEmptySub(directory, recFile)


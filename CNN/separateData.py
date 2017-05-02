import os
from shutil import copy
import random

def separation(readFolder, targetFolder, train_num, test_num):
	# imgNames = [img for img in os.listdir(readFolder) if img.endswith('.png')]
	# get test data
	# testImgNames = random.sample(imgNames,test_num)
	# for imgName in testImgNames:
	# 	srcPath = os.path.join(readFolder, imgName)
	# 	copy(srcPath, targetFolder)
	# 	os.remove(srcPath)
	# modify train data
	imgNames = [img for img in os.listdir(readFolder) if img.endswith('.png')]
	exist_img = len(imgNames)
	if exist_img > train_num:
		trainImgNames = random.sample(imgNames,exist_img - train_num)
		for imgName in trainImgNames:
			imgPath = os.path.join(readFolder, imgName)
			os.remove(imgPath)


readFolder = '../Fault_Horiz'
targetFolder = '../TestMAGE_GROUP_HORIZ_FAULT'
train_num = 7000
test_num = 353

separation(readFolder, targetFolder, train_num, test_num)

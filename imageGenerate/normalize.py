# NORMALIZE SYMBOL IMAGES
# Script parameters: folderName

import os
from resize import resizeImg

# RECURSIVELY PRE-PROCESS
def normImg(readPath, savePath, norm_size):
	if os.path.isdir(readPath):
		if not os.path.exists(savePath):
			os.makedirs(savePath)
		contents = os.listdir(readPath)
		for con in contents:
			if con != 'wholeImgs' and not con.startswith('.'):
				nextReadPath = os.path.join(readPath, con)
				nextSavePath = os.path.join(savePath, con)
				normImg(nextReadPath, nextSavePath, norm_size)
	elif os.path.isfile(readPath):
		resizeImg(readPath, savePath, norm_size)


####### TRAINNING IMAGES ########

# paras
# folderName = 'MfrDB'
# readPath = 'TrainIMAGE/'+ folderName
# savePath = 'TrainIMAGE_NORM/' + folderName
# norm_size = 32
# normImg(readPath, savePath, norm_size)


####### TESTING IMAGES ########

# paras
# readPath = 'TestIMAGE'
# savePath = 'TestIMAGE_NORM'
# norm_size = 32
# normImg(readPath, savePath, norm_size)


####### TEST SINGLE IMAGE ########

# fileName = '2_MfrDB0001_0.png'
# readImg = 'testImg/2/' + fileName
# savePath = 'testImg/' + fileName
# norm_size = 32
# resizeImg(readImg, savePath, norm_size)

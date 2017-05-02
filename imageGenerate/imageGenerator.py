# Generate images out from inkml file
# Script parameters: folderName

import interpretInkml
import os

####### TRAINNING IMAGES GENERATION #######

# mode = 'Normal'
# folderName = 'extension'
# saveDir = '../TrainIMAGE_RAW/' + folderName
# readDir = '../inkml/TrainINKML/' + folderName
# wholeImgFolder = 'wholeImgs'
# print(folderName)

# if not os.path.exists(saveDir):
# 	os.makedirs(saveDir)
# wholeImgDir = saveDir+'/'+wholeImgFolder
# if not os.path.exists(wholeImgDir):
# 	os.makedirs(wholeImgDir)

# files = os.listdir(readDir)
# idx = 1
# for f in files:
# 	if f.endswith('.inkml'):
# 		print("Start Image " + str(idx) + ": " + f + '  ')
# 		filePath = os.path.join(readDir,f)
# 		interpretInkml.extract_images(filePath, saveDir, wholeImgFolder, mode)
# 		print("Complete!")
# 		idx = idx+1


####### TEST IMAGES GENERATION #######

# mode = 'Normal'
# saveDir = 'TestIMAGE'
# readDir = 'TestINKML'
# wholeImgFolder = 'wholeImgs'

# if not os.path.exists(saveDir):
# 	os.makedirs(saveDir)
# wholeImgDir = saveDir+'/'+wholeImgFolder
# if not os.path.exists(wholeImgDir):
# 	os.makedirs(wholeImgDir)

# files = os.listdir(readDir)
# idx = 1
# for f in files:
# 	if f.endswith('.inkml'):
# 		print("Start Image " + str(idx) + ": " + f + '  ')
# 		filePath = os.path.join(readDir,f)
# 		interpretInkml.extract_images(filePath, saveDir, wholeImgFolder, mode)
# 		print("Complete!")
# 		idx = idx+1


####### TEST SINGEL IMAGE #######

mode = 'Normal'
filePath = '../inkml/TrainINKML/MfrDB/MfrDB0001.inkml'
saveDir = 'testImg'
wholeImgFolder = 'wholeImgs'

if not os.path.exists(saveDir):
	os.makedirs(saveDir)
wholeImgDir = saveDir+'/'+wholeImgFolder
if not os.path.exists(wholeImgDir):
	os.makedirs(wholeImgDir)
interpretInkml.extract_images(filePath, saveDir, wholeImgFolder, mode)

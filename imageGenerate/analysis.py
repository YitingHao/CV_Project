# ANYALYZE ALL SUB-IMAGES

import os

def distribution(folderPath, recFile, thresh1, thresh2):
	f = open(recFile, 'wr+')
	list1 = []
	list2 = []
	symbols = [sym for sym in os.listdir(folderPath) if not sym.startswith('.')]
	for sym in symbols:
		symPath = os.path.join(folderPath, sym)
		imgs = [img for img in os.listdir(symPath) if img.endswith('.png')]
		f.write(sym + ': ' + str(len(imgs)) + '\n')
		size = len(imgs)
		if size < thresh1:
			list1.append((sym, size))
		elif size < thresh2:
			list2.append((sym, size))
	# output
	f.write('\n\n' + str(thresh1) + ':\n')
	for sym_tup in list1:
		f.write(sym_tup[0] + '  ' + str(sym_tup[1]) + '\n')
	f.write('\n\n[' + str(thresh1) + ', ' + str(thresh2) + ']:\n')
	for sym_tup in list2:
		f.write(sym_tup[0] + '  ' + str(sym_tup[1]) + '\n')
	f.close()

folder = '../TestIMAGE_NORM'
recFile = 'anaysis_test.txt'
distribution(folder, recFile, -1, 1)

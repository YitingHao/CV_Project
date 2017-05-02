
import os
import re

def parseSym(inkmlPath):
	# regex setting
	group_pattern = '^<traceGroup xml:id="(.*)">$'
	symbol_pattern = '^<annotation type="truth">(.*)<\/annotation>$'
	gp_regex = re.compile(group_pattern)
	sym_regex = re.compile(symbol_pattern)

	start_gp = False
	symbols = []

	with open(inkmlPath) as readf:
		for line in readf:
			line = line.strip()
			if start_gp:
				sym_res = sym_regex.match(line)
				if sym_res:
					symbol = sym_res.group(1).replace("\\","")
					if len(symbol) == 1 and symbol.isupper():
						symbol = 'upper_' + symbol
					symbols.append(symbol)
			elif gp_regex.match(line):
				start_gp = True

	del symbols[0]
	symbols = sorted(symbols)
	return ' '.join(symbols)

folder = '../inkml/TestINKML'
inkmlFiles = [f for f in os.listdir(folder) if f.endswith('.inkml')]

standardFile = 'symbol_standard.txt'
f = open(standardFile,'wr+')
for inkmlFile in inkmlFiles:
	inkmlPath = os.path.join(folder, inkmlFile)
	symbStr = parseSym(inkmlPath)
	f.write(inkmlFile.replace('.inkml','.png') + ":" + symbStr + '\n')
f.close()





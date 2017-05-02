
import editdistance

predFile = 'predictSymbols.txt'
standFile = 'symbol_standard.txt'

predict_dict = {}
standard_dict = {}

maxSymLen = 0

# get dictionary
with open(predFile, 'r') as predF:
	for line in predF:
		k_v_split = line.strip().split(':')
		key = k_v_split[0]
		syms = sorted(k_v_split[1].split())
		predict_dict[key] = syms
		maxSymLen = max(maxSymLen, len(syms))

with open(standFile, 'r') as standF:
	for line in standF:
		k_v_split = line.strip().split(':')
		key = k_v_split[0]
		syms = k_v_split[1].split()
		standard_dict[key] = syms

dis = 0
length = 0

dis1 = 0 
len1 = 0

dis2 = 0
len2 = 0

dis3 = 0
len3 = 0

dis4 = 0
len4 = 0

for img, syms in predict_dict.iteritems():
	cur_dis = editdistance.eval(syms, standard_dict[img])
	cur_syms = len(standard_dict[img])
	dis += cur_dis
	length += cur_syms
	if len(syms) <= 6:
		dis1 += cur_dis
		len1 += 1
	elif len(syms) <= 12:
		dis2 += cur_dis
		len2 += 1
	elif len(syms) <= 18:
		dis3 += cur_dis
		len3 += 1
	else:
		dis4 += cur_dis
		len4 += 1


print maxSymLen

print 'total distance:', dis
print 'total number of symbols:', length
print 'average number of symbols:', float(length) / len(predict_dict)
print 'averga distance:', float(dis) / len(predict_dict)

print float(dis1) / len1
print float(dis2) / len2
print float(dis3) / len3
print float(dis4) / len4


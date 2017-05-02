
import re
import matplotlib.pyplot as plt

# folder = 'sym_error.txt'
# title = 'CNN Symbol Classifier: Epoch vs Error'

# folder = 'vertic_error.txt'
# title = 'CNN Vertical Grouping Binary Classifier: Epoch vs Error'

folder = 'horiz_error.txt'
title = 'CNN Horizontal Grouping Binary Classifier: Epoch vs Error'

# regex setting
epoch_pattern = '^Step (.*) \(epoch (.*)\), (.*) ms$'
error_pattern = '^Validation error: (.*)%$'

epoch_regex = re.compile(epoch_pattern)
error_regex = re.compile(error_pattern)

x = []
y = []

with open(folder, 'r') as f:
	for line in f:
		line = line.strip()
		epoch_res = epoch_regex.match(line)
		if epoch_res:
			x.append(float(epoch_res.group(2)))
			continue
		error_res = error_regex.match(line)
		if error_res:
			y.append(float(error_res.group(1)))

plt.plot(x, y, 'r--', linewidth=2.0)
plt.axis([0, 4, 0, 51])
plt.xlabel('Number of Epoch')
plt.ylabel('Validation Error (%)')
plt.title(title)
plt.show()


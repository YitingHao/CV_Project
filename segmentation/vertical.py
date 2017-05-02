
import os
from collections import defaultdict
import numpy as np
from skimage import io, measure
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# GROUP VERTICALLY
def vertical_merge(imgFolder, imgName):
	NUM_VERTICAL_COMB = 2
	NUM_HORIZONTAL_COMB = 2
	VOL_V_THRESHOLD_RATIO = 1.5
	# VOL_H_THRESHOLD_RATIO = 0.1
	V_THRESHOLD_RATIO_JUDGE = 0.33
	H_THRESHOLD_RATIO_JUDGE = 1
	H_THRESHOLD_RATIO_HELP = 1
	V_THRESHOLD_RATIO_HELP = 0.6
	# grayscale
	imgPath = os.path.join(imgFolder, imgName)
	img = rgb2gray(io.imread(imgPath))
	img_size = img.shape
	thresh = threshold_mean(img)
	# img = (img > thresh) * 255
	img = img > thresh * 0.8	#### ATTENTION!
	# connected components
	[label, num_label] = measure.label(img, neighbors=8, background=1, return_num = True)
	CCs = measure.regionprops(label)
	# controids, horizontals, verticals, heights, vols
	centroids = []
	cc_heights = []
	cc_areas = []
	for cc in CCs:
		cc_heights.append(cc.bbox[2] - cc.bbox[0])
		cc_areas.append(cc.area)
		centroids.append(cc.centroid)
	horizontals = np.array([centroid[0] for centroid in centroids])
	verticals = np.array([centroid[1] for centroid in centroids])
	cc_heights = np.array(cc_heights)
	cc_areas = np.array(cc_areas)
	# get character heights, get character areas for later criteria
	mean_char_height = np.mean(cc_heights)
	mean_char_area = np.mean(cc_areas)
	if len(cc_heights) > 2:
		kmeans = KMeans(n_clusters=2).fit([(h,0) for h in sorted(cc_heights)])
		ch_idxs = np.nonzero(kmeans.labels_ == kmeans.labels_[-1])[0]
		mean_char_height = np.mean(np.array(sorted(cc_heights))[ch_idxs])
		kmeans = KMeans(n_clusters=2).fit([(h,0) for h in sorted(cc_areas)])
		ch_idxs = np.nonzero(kmeans.labels_ == kmeans.labels_[-1])[0]
		mean_char_area = np.mean(np.array(sorted(cc_areas))[ch_idxs])
	# get mean vertical distance, set H_THRESHOLD_RATIO_HELP
	v_dis = []
	for i in xrange(num_label):
		for j in xrange(i+1, num_label):
			v_dis.append(abs(horizontals[i] - horizontals[j]))
	v_dis = np.array(v_dis)
	if len(v_dis) > 6 and np.amax(v_dis) * 0.8 < mean_char_height * 1.2:
		# more than 4 CCs
		H_THRESHOLD_RATIO_HELP = 2
		
	v_idx = np.argsort(verticals)
	v_points_sort = verticals[v_idx]
	# get threshold
	diff_v = []
	for i in xrange(1,num_label):
		diff_v.append(v_points_sort[i] - v_points_sort[i-1])
	mean_div_v = np.mean(diff_v)
	threshold = mean_div_v * V_THRESHOLD_RATIO_JUDGE
	verti_comb = defaultdict(list)
	for i in xrange(num_label):
		for j in xrange(num_label):
			if i != j and abs(verticals[i] - verticals[j]) < threshold:
				verti_comb[i].append(j)
	# modify dict: delete elements (when more then 3) in the list 
	#			   based on horizontal distances
	for cc_idx, verti_list in verti_comb.iteritems():
		if len(verti_list) > NUM_VERTICAL_COMB:
			horiz_diff = [abs(horizontals[v_idx]-horizontals[cc_idx]) \
						  for v_idx in verti_list]
			selected_idx = np.argsort(horiz_diff)
			# print selected_idx
			verti_comb[cc_idx] = [verti_list[idx] \
								  for idx in selected_idx[:NUM_VERTICAL_COMB]]
	# print verti_comb
	# groups tuples
	groups = set()
	for cc_idx, verti_list in verti_comb.iteritems():
		delete_list = []
		for bi_idx in verti_list:
			if abs(horizontals[cc_idx] - horizontals[bi_idx]) < mean_char_height * H_THRESHOLD_RATIO_HELP \
			and CCs[cc_idx].area + CCs[bi_idx].area < mean_char_area * VOL_V_THRESHOLD_RATIO:
				groups.add(tuple(sorted([cc_idx, bi_idx])))
			else:
				delete_list.append(bi_idx)
		for idx in delete_list:
			verti_list.remove(idx)
		if len(verti_list) == 2:
			groups.add(tuple(sorted([cc_idx, verti_list[0], verti_list[1]])))
	# print groups
	return label, CCs, groups

# # GROUP HORIZONTALLY
# def horizontal_merge(imgFolder, imgName):
# 	# grayscale
# 	imgPath = os.path.join(imgFolder, imgName)
# 	img = rgb2gray(io.imread(imgPath))
# 	img_size = img.shape
# 	thresh = threshold_mean(img)
# 	# img = (img > thresh) * 255
# 	img = img > thresh * 0.8		# ATTENTION!!!
# 	# connected components
# 	[label, num_label] = measure.label(img, neighbors=8, background=1, return_num = True)
# 	CCs = measure.regionprops(label)
# 	# controids, horizontals, verticals, heights, vols
# 	centroids = []
# 	for cc in CCs:
# 		centroids.append(cc.centroid)
# 	horizontals = np.array([centroid[0] for centroid in centroids])
# 	verticals = np.array([centroid[1] for centroid in centroids])

# 	h_idx = np.argsort(horizontals)
# 	h_points_sort = horizontals[h_idx]
# 	print horizontals
# 	print h_points_sort

# 	# get threshold
# 	diff_h = []
# 	for i in xrange(1,num_label):
# 		diff_h.append(h_points_sort[i] - h_points_sort[i-1])
# 	mean_div_h = np.mean(diff_h)
# 	threshold = mean_div_h * H_THRESHOLD_RATIO_JUDGE
# 	print threshold

# 	# horizontal distance
# 	h_mean_disc = 0
# 	for i in xrange(num_label):
# 		for j in xrange(i+1, num_label):
# 			h_mean_disc += abs(verticals[i] - verticals[j])
# 	if num_label > 2:
# 		h_mean_disc /= num_label * (num_label-1) / 2

# 	# generate dict: cc_idx -> list of vertically close cc_idxes
# 	horizon_comb = defaultdict(list)
# 	for i in xrange(num_label):
# 		for j in xrange(num_label):
# 			if i != j and abs(horizontals[i] - horizontals[j]) < threshold:
# 				horizon_comb[i].append(j)
# 	print horizon_comb

# 	# modify dict: delete elements (when more then 3) in the list 
# 	#			   based on horizontal distances
# 	for cc_idx, horizon_list in horizon_comb.iteritems():
# 		if len(horizon_list) > NUM_HORIZONTAL_COMB:
# 			verti_diff = [abs(verticals[h_idx]-verticals[cc_idx]) \
# 						  for h_idx in horizon_list]
# 			print verti_diff
# 			selected_idx = np.argsort(verti_diff)
# 			print selected_idx
# 			horizon_comb[cc_idx] = [horizon_list[idx] \
# 								  for idx in selected_idx[:NUM_HORIZONTAL_COMB]]
# 	print horizon_comb

# 	groups = set()
# 	for cc_idx, selected_idx in horizon_comb.iteritems():
# 		delete_list = []
# 		selfArea = CCs[cc_idx].area
# 		for bi_idx in selected_idx:
# 			if abs(verticals[cc_idx] - verticals[bi_idx]) < h_mean_disc * V_THRESHOLD_RATIO_HELP:
# 				groups.add(tuple(sorted([cc_idx, bi_idx])))
# 			else:
# 				delete_list.append(bi_idx)
# 		for idx in delete_list:
# 			selected_idx.remove(idx)
# 		if len(selected_idx) == 2:
# 			groups.add(tuple(sorted([cc_idx, selected_idx[0], selected_idx[1]])))

# 	print groups

# 	colors_seq = ['g','b','y','k','w']
# 	color_idx = 0
# 	fig,ax = plt.subplots(1)
# 	ax.imshow(img)
# 	for cc in CCs:
# 		print(cc.bbox)
# 		ax.add_patch(patches.Rectangle((cc.bbox[1], cc.bbox[0]), \
# 									   cc.bbox[3] - cc.bbox[1], \
# 									   cc.bbox[2] - cc.bbox[0], \
# 									   fill = False, \
# 									   edgecolor=colors_seq[color_idx % len(colors_seq)]))
# 		color_idx += 1
# 	plt.show()

# 	return label, CCs, groups

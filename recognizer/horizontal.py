from skimage import measure
from skimage.draw import line
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.misc import imresize
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
import matplotlib.patches as patches

norm_size = 32

def gene_image(coords):
    coords_x = [c[0] for c in coords]
    coords_y = [c[1] for c in coords]
    x_min = min(coords_x)
    x_max = max(coords_x)
    y_min = min(coords_y)
    y_max = max(coords_y)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    im = np.ones((width, height))
    coords_num = len(coords_x)
    for i in range(coords_num):
        im[coords_x[i]-x_min, coords_y[i]-y_min] = 0
    return im

def parse_dict(gdict):
    
    num = []
    for key in gdict:
        gstr = gdict[key]
        num = num + gstr.split(',')
        num = [int(x) for x in num]

    num = list(set(num))
    return num
        

def merge_left_right_symbols(coords, boxes):
    
    
    coords_num = len(coords)
    w_list = np.zeros((1, coords_num))
    h_list = np.zeros((1, coords_num))
    coords_new = []
    boxes_new = []
    ratio_w = 0.8
    ratio_h = 1.1

    for i in range(coords_num):
        y_min = boxes[i][0]
        x_min = boxes[i][1]
        y_max = boxes[i][2]
        x_max = boxes[i][3]
        w_list[0][i] = x_max - x_min
        h_list[0][i] = y_max - y_min
    w_average = np.sum(w_list, axis = 1)/coords_num
    h_average = np.sum(h_list, axis = 1)/coords_num
    
    dist_x = math.pow(ratio_w * w_average, 2)
    dist_y = math.pow(ratio_h * h_average, 2)
    dist_thr = math.sqrt(dist_x + dist_y)
    dist_matrix = calc_dist_matrix(coords)
    
    dist = 0
    
    group_dict = {}


    for i in range(coords_num):
        box_i = boxes[i]
        index = [dist_matrix[i].index(d) for d in  dist_matrix[i] if d<dist_thr]
        for j in index:
            box_j = boxes[j]
            i_x_min = box_i[1]
            i_x_max = box_i[3]
            i_y_min = box_i[0]
            i_y_max = box_i[2]
            
            j_x_min = box_j[1]
            j_x_max = box_j[3]
            j_y_min = box_j[0]
            j_y_max = box_j[2]

            jidist = j_x_min - i_x_max
            ijdist = i_x_min - j_x_max
            
            if jidist > 0:
                dist = jidist
            else:
                dist = ijdist
            
            if i_y_min < j_y_max and i_y_max > j_y_min and dist < ratio_w * w_average:
                boxes_new.append((min(i_y_min, j_y_min), min(i_x_min, j_x_min), max(i_y_max, j_y_max), max(i_x_max, j_x_max)))
                coords_new.append(np.concatenate((coords[i], coords[j])))
                group_dict[str(len(group_dict))] = str(i)+','+str(j)
                
    return coords_new, boxes_new, group_dict

def find_centroid(coord):


    sum_x = sum(coord[:,0])
    sum_y = sum(coord[:,1])
    area = len(coord)
    c_x = sum_x/area
    c_y = sum_y/area
    return c_x, c_y

def calc_dist_matrix(coords):
    coords_num = len(coords)
    dist_matrix = []

    for i in range(coords_num):
        dist_matrix.append([])
        [c_i_x, c_i_y] = find_centroid(coords[i])
        for j in range(coords_num):

            [c_j_x, c_j_y] = find_centroid(coords[j])
            dist_x = math.pow((c_i_x - c_j_x), 2)
            dist_y = math.pow((c_i_y - c_j_y), 2)
            dist_i_j = math.sqrt((dist_x + dist_y))
            
            if j <= i:
                dist_matrix[i].append('Inf')
            else:
                dist_matrix[i].append(dist_i_j)

    return dist_matrix


def merge_overlap(coords, boxes, gdict):
    
    group_dict = gdict
    coords_new = coords
    boxes_new = boxes
    
    group_num = len(gdict)
    
    for i in range(group_num):
        box_i = boxes[i]
        gstr_i = gdict[str(i)]
        gids_i = gstr_i.split(',')
        for j in range(i+1, group_num):
            box_j = boxes[j]
            i_x_min = box_i[1]
            i_x_max = box_i[3]
            i_y_min = box_i[0]
            i_y_max = box_i[2]
            
            j_x_min = box_j[1]
            j_x_max = box_j[3]
            j_y_min = box_j[0]
            j_y_max = box_j[2]
            
            gstr_j = gdict[str(j)]
            gids_j = gstr_j.split(',')
            gids_same = set(gids_i) & set(gids_j)
            if len(gids_same) != 0:
                boxes_new.append((min(i_y_min, j_y_min), min(i_x_min, j_x_min),\
                                  max(i_y_max, j_y_max), max(i_x_max, j_x_max)))
                coords_new.append(np.concatenate((coords[i], coords[j])))
                group_dict[str(len(group_dict))] = gdict[str(i)]+','+gdict[str(j)]
            
    return coords_new, boxes_new, group_dict

# RESIZE IMAGE
def resizeImg(im, norm_size):
	img = im
	img_size = img.shape
	padding = max(img_size) - min(img_size)
	pad_s = padding / 2
	pad_l = padding - pad_s
	if img_size[0] == max(img_size):
		img = np.lib.pad(img, ((0,0),(pad_s,pad_l)), \
						'constant', constant_values=((1,1),(1,1)))
	else:
		img = np.lib.pad(img, ((pad_s,pad_l),(0,0)), \
						'constant', constant_values=((1,1),(1,1)))
	img = imresize(img, (norm_size, norm_size))
	thresh = np.mean(img)
	img = (img > thresh) * 255
	return img

# CHECK IF OUR GROUP ARE RIGHT OR NOT FROM probs RETURNED FROM CNN
'''
def group_results(gdict, probs):
    thr = 0.5
    gdict_fail = {}
    gdict_succ = {}
    for i in range(len(probs)):
        if prob[i] < thr:
            gdict_fail(str(len(gdict_fail))) = gdict[str(i)]
        else:
            gdict_succ(str(len(gdict_succ))) = gdict[str(i)]
    for i in range(len(gdict_succ)):
        for j in range(i+1, len(gdict_succ)):
            
 '''           
def dict2list(gdict):
    gdict_new = {}
    for key in gdict:
        gstr = gdict[key]
        num = gstr.split(',')
        num = [int(x) for x in num]
        gdict_new[key] = list(set(num))
    return gdict_new


def horizontal(label):
    
    # # grayscale
    # imgPath = os.path.join(imgFolder, imgName)
    # img = rgb2gray(io.imread(imgPath))
    # img_size = img.shape
    # thresh = threshold_mean(img)
    # # img = (img > thresh) * 255
    # img = img > thresh * 0.8    #### ATTENTION!
    # # connected components
    # [label, num_label] = measure.label(img, neighbors=8, background=1, return_num = True)
    CCs = measure.regionprops(label)

    co1 = []
    box1 = []
    box_list = range(len(CCs))
            
    for i in box_list:
        co1.append(CCs[i].coords)
        box1.append(CCs[i].bbox)

    [co2,box2,gdict1] = merge_left_right_symbols(co1, box1)   

    [co3,box3,gdict2] = merge_overlap(co2, box2, gdict1)
            
    num_list = parse_dict(gdict2)
    for num in num_list:
        box_list.remove(num)

    gdict = dict2list(gdict2)

    groups = set()
    for key, value in gdict.iteritems():
        groups.add(tuple(sorted(value)))
    groups = list(groups)

    # colors_seq = ['g','b','y','k','w']
    # color_idx = 0
    # fig,ax = plt.subplots(1)
    # ax.imshow(img)
    # for cc in CCs:
    #     print(cc.bbox)
    #     ax.add_patch(patches.Rectangle((cc.bbox[1], cc.bbox[0]), \
    #                                    cc.bbox[3] - cc.bbox[1], \
    #                                    cc.bbox[2] - cc.bbox[0], \
    #                                    fill = False, \
    #                                    edgecolor=colors_seq[color_idx % len(colors_seq)]))
    #     color_idx += 1
    # plt.show()

    return label, CCs, groups


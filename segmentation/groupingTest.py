import vertical
import os
import horizontal

imgFolder = '../TestWhole'

# imgs = [img for img in os.listdir(imgFolder) if img.endswith('.png')]
# for img in imgs:
# 	print img
# 	group.horizontal_merge(imgFolder, img)

#### Single Image ####
# img = 'rit_42125_1.png'
img = 'rit_42125_4.png'
horizontal.horizontal(imgFolder, img)

# colors_seq = ['g','b','y','k','w']
# coding:utf-8

## 生成webface的训练list，参照sphereface的matlab样本准备，去处了与LFW重复的17对编号，参照happynear的facedata


import numpy as np
import sys,os

#dir_root = '/home/sdc1/zf/datasets/face-recognition/CASIA-WebFace/'
dir_root = 'D:/others/FacialExpressionImage/CASIA-WebFace/'
img_root = dir_root + 'CASIA-WebFace/'
img_list = dir_root + 'list.txt'

overlap_name = ['0000513','0004539','0004763','0005009','0005082','0005172','0166921','0208962','0430107'
,'0662519','0955471','1056413','1091782','1193098','1303492','2425974','3478560']
list = []
files = os.listdir(img_root)
fid = open(img_list,'w')
cnt = 0
for file in files:
    m = os.path.join(img_root,file)
    if(os.path.isdir(m)):
        h = os.path.split(m)
        if(h[1] in overlap_name):
            continue
        img_name = os.listdir(m)
        for im in img_name:
            fileName = os.path.join(m,im)
            if (os.path.isdir(fileName)):
                continue
            fid.write('%s %d\n' %(fileName,cnt))
        cnt = cnt+1
        list.append(h[1])
fid.close()

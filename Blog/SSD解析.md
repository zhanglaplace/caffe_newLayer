---
title: SSD算法与Caffe实现解析
date: 2017-10-30 14:16:01
tags: [Facial Expression,DeepLearning,Face Recognition]
categories: Facial Expression
---


  本文主要介绍SSD算法，并结合具体的Prototxt完成必要的说明，如有错误，欢迎指正。SSD为近两年来提出的one-stage中非常优秀的一种目标检测算法，特点很鲜明，利用多特征的融合完成目标的定为与识别。

  论文:*Single Shot MultiBox Detector* \
  intro: ECCV 2016 Oral
  arxiv: http://arxiv.org/abs/1512.02325 \
  paper: http://www.cs.unc.edu/~wliu/papers/ssd.pdf \
  slides: http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf \
  github(Official): https://github.com/weiliu89/caffe/tree/ssd \
  video: http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973 \
  github: https://github.com/zhreshold/mxnet-ssd \
  github: https://github.com/zhreshold/mxnet-ssd.cpp \
  github: https://github.com/rykov8/ssd_keras \
  github: https://github.com/balancap/SSD-Tensorflow \
  github: https://github.com/amdegroot/ssd.pytorch \
  github(Caffe): https://github.com/chuanqi305/MobileNet-SSD

# 算法概述
  SSD算法是一种直接预测bbox和类别confidence的目标检测方法，相对于RCNN类型的网络来说，省去了proposals的生成过程，因此在速度上能够有较大的提高。算法对VGGG16网络做了基本的修改，去掉了最后的两个fc全连接层，转而修改为全连接层，并在后续添加了4个卷积层，对其中五个卷积层的feature map与两个3*3的卷积核进行卷积，1个输出分类的confidence(20+1)，另外一个输出回归的location(x,y,w,h)，同时这五个卷积层通过priorBox层生成defaultBox，每层每层defaultBox事先给定，最后整合所有五个卷积层输出的类别与候选区域，计算loss。
  算法结果：对于300\*300的图像，SSD可以在VOC2007 test上有74.3%的mAP，速度是59 FPS(Titan X)，对于512\*512的输入, SSD可以有76.9%的mAP。相比之下Faster RCNN是73.2%的mAP和7FPS，YOLO是63.4%的mAP和45FPS。即便对于分辨率较低的输入也能取得较高的准确率。可见作者并非像传统的做法一样以牺牲准确率的方式来提高检测速度.

# 算法详解
  SSD在训练的时候需要准备图像以及每个图像内真实的bbox，即Ground Truth boxes。
  利用了upper和lower的feature Map做检测。如图1所示：在每一个提取特征的featuremap上，每一个像素点可以认为是

---
title: Caffe VisionLayer分析
date: 2017-10-24 19:16:01
tags: [Caffe,DeepLearning]
categories: Caffe
---

### Caffe VisionLayer
  老版本中Caffe有$VisionLayer$,其中主要包含了卷积层，采样层，$im2col$层等，本文将结合自己的理解对这些层次进行分析，在自己学习总结的同事，写下对源码的理解。

<!--more-->
#### $(1) \, im2colLayer$
  为了提高$conv$计算的速度，$caffe$采取了$im2col$的方式，通过对滤波器$kernel$和$feature map$做形式上的改变，从而达到提高计算的作用,因此在进行$im2col$前，必须要知道$kernel$的尺度，卷积方式，输入输出的通道数目以及$batch\_size$的大小。
1.基本数据成员

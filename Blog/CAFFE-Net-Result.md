---
title: Caffe经典网络-cifar10验证
date: 2017-11-04 11:44
tags: [Facial Expression,DeepLearning,Face Recognition]
categories: Facial Expression
---

# 概述
  本文对经典的网络模型在$cifar10$上进行了大致的验证，对比数据包括占用的显存，测试的准确率，训练时间、测试时间以及模型大小进行对比。由于部分网络的显存占用大，因此无法保证batch_size的设置一致，电脑显卡为$GTX1080Ti$，$11G$显存，迭代次数均为64000，初步的实验结果如下表:

|     modelStrut     | Gpu Memory | Batch_size | Accuracy | model Size | train time | test time |
|:------------------:|:----------:|:----------:|:--------:|:----------:|:----------:| --------- |
|        Alex        |   ~0.5G    |  128+128   |  0.7243  |    3.0M    |   ~7min    |           |
|       LeNet        |   ~0.5G    |  128+128   |  0.7840  |   ~0.35M   |   ~28min   |           |
|      BN-LeNet      |   ~1.35G   |  128+128   |  0.7985  |   ~0.35M   |   ~16min   |           |
|       VGG16        |   ~1.3G    |            |          |            |            |           |
|   SqeezeNet_v1.1   |   ~1.25G   |  128+128   |  0.8030  |   ~2.8M    |   ~30min   |           |
|      ResNet20      |   ~1.5G    |  128+128   |  0.8303  |   ~1.1M    |  ~1h5min   |           |
|      ResNet32      |   ~2.5G    |  128+128   |  0.8741  |   ~1.9M    |  ~1h53min  |           |
|      ResNet56      |            |            |          |            |            |           |
|      WRN28_10      |            |            |          |            |            |           |
|      Dense30       |            |            |          |            |            |           |
|        NIN         |   ~1.5G    |  128+128   |  0.8411  |    ~25M    |   ~46min   |           |
|     GoolgeNet      |   ~1.3G    |  128+128   |  0.7865  |    ~25M    |   ~30min   |           |
| Ourself(28residual |            |            |          |            |            |           |

说明：由于训练的时候节省时间，有的网络一起开始训练的，因此显存占用与训练时间可能与单个测试的时间有差异

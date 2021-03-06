---
title: Caffe经典网络-cifar10验证
date: 2017-11-04 11:44
tags: [Facial Expression,DeepLearning,Face Recognition]
categories: Facial Expression
---

# 概述
  本文对经典的网络模型在$cifar10$上进行了大致的验证，对比数据包括占用的显存，测试的准确率，训练时间、测试时间以及模型大小进行对比。由于部分网络的显存占用大，因此无法保证batch_size的设置一致，电脑显卡为$GTX1080Ti$，$11G$显存，迭代次数均为64000，初步的实验结果如下表:

|   modelStrut   | Gpu Memory | Batch_size | Accuracy | model Size | train time | test time |
|:--------------:|:----------:|:----------:|:--------:|:----------:|:----------:| --------- |
|      Alex      |   ~0.5G    |  128+128   |  0.7243  |    3.0M    |   ~7min    | ~0.67ms   |
|     LeNet      |   ~0.5G    |  128+128   |  0.7840  |   ~0.35M   |   ~28min   | ~0.84ms   |
|    BN-LeNet    |   ~1.35G   |  128+128   |  0.7985  |   ~0.35M   |   ~16min   | ~0.86ms   |
| SqeezeNet_v1.1 |   ~1.25G   |  128+128   |  0.8030  |   ~2.8M    |   ~30min   | ~3.02ms   |
|    ResNet20    |   ~1.5G    |  128+128   |  0.8303  |   ~1.1M    |  ~1h5min   | ~4.55ms   |
|    ResNet32    |   ~2.5G    |  128+128   |  0.8741  |   ~1.9M    |  ~1h53min  | ~6.91ms   |
|    ResNet56    |   ~4.0G    |  128+128   |  0.8830  |   ~3.4M    |  ~3h20min  | ~11.65ms  |
|    WRN28_10    |   ~10.5G   |   64+128   |  0.8905  |   ~140M    | ~12h55min  | ~9.31ms   |
|    Dense30     |   ~10.0G   |   32+64    |  0.9195  |   ~4.0M    |  ~3h38min  | ~14.73ms  |
|      NIN       |   ~1.5G    |  128+128   |  0.8411  |    ~25M    |   ~46min   | ~1.99ms   |
|   GoolgeNet    |   ~1.3G    |  128+128   |  0.7865  |    ~25M    |   ~30min   | ~8.42ms   |

说明：由于训练的时候节省时间，有的网络一起开始训练的，因此显存占用与训练，测试时间可能与单个测试的时间有差异.

| modelStruct   | Gpu Memory | Batch_size | mAp    | model_size | train_time | test time |
| ------------- | ---------- | ---------- | ------ | ---------- | ---------- | --------- |
| MobileNet SSD | ~7.4G      | 24+8       | 0.7243 | ~23M       | 25k--~17h  | ~60ms     |
| faceBox       | ~7.7G      | 8+4        |        |            |            |           |

##

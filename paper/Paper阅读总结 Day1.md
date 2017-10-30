---
title: Paper阅读总结Day1
date: 2017-10-30 14:16:01
tags: [Facial Expression,DeepLearning,Face Recognition]
categories: Facial Expression
---

# Paper阅读总结Day1

## Convolutional Neural Networks For Facial Expression Recognition

### 文章思想
  简单的一篇关于表情识别的文章，运用简单的CNN结构，在文章中对比了深层次的网络结构和浅层次的网络结构的效果，同时将前向的最后一层特征与自己手动提取的Hog特征做了特征融合，并重新训练一个全连接层，得到的效果与不用特征融合效果一致。

### 文章使用数据集
  Fer2013 Database，通过浅层次和深层次的横向对比与 加入hog与不加hog的横向对比

### 实验效果与结论
  深层次的CNN准确率大概是65%，加入HOG与不加效果基本一致，结论是否定了Hog特征融合对表情识别有效果的提升。


## Island Loss for Learning Discriminative Features in Facial Expression Recognition

### 文章思想
  简单的在centerLoss的基础上，添加了衡量各类别类心的loss，由于centerloss只关注了样本到类心的类内距离，而IslandLoss在关注类心距离的同时，添加了类间距离的loss，采用余弦距离衡量类心的相似程度。
$$\zeta = \zeta_S+\lambda (\zeta_C+\lambda_1\zeta_{is})$$

### 文章使用的数据集
  Oulu-CASIA database 、 Extended Cokn-kanada和MMI database，fer2013

### 实验效果与结论
  在各个数据集上的表现都优于SoftmaxLoss以及 CenterLoss+SoftmaxLoss.需要把握各个loss的权重调节

## End to End Deep Learning for Single Step Real-Time Facial Expression Recognition

### 文章思想
  实现一个集合人脸检测与人脸表情分类的一体的网络---Faster-RCNN。替换了Faster-RCNN前面的预训练的网络结构，采取了VGG16和ResNet50，做对比后VGG可以达到10fps，ResNet50-5fps，感觉略有水分。
### 文章使用数据集
  Extended Cokn-kanada 和 FER2013
### 实验效果与结论
  能够在CK+上10折达到94.7的Accuracy 10fps 实际使用基本不可能，RPN的人脸检测稳定性很低

## Comparative Study of Human Age Estimation Based on Hand-Crafted and Deep Face Features

### 文章思想
  自己提取的特征(LBP、Hog、BSIF)以后CNNs提取的特征(VGG-face、Image-VGG-F、VGG16、DEX-IMDB-WIKI and DEX-ChaLearn-ICCV2015 Features)，五个CNNs网络，有包含图像分类，人脸识别，目标检测与年龄预估。实际上就是特征做融合，然后用PLS regression 偏最小二乘法回归分析。
### 文章使用数据集
  MORPH和PAL database
### 实验效果与结论
  实验对比了几种特征单独的实验效果以及crop后的效果，实验说明了最后的回归很重要，然后CNN的特征比这些自己提取的特征好。

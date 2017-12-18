---
title: FaceRecgnition-CenterLoss
date: 2017-11-04 11:44
tags: [DeepLearning,Face Recognition]
categories: Face Recognition
---

  由于CNN的发展，最近几年faceRecognition(FR)方面，发展迅速，不同的loss已经embedding策略相继提出，本文主要介绍较为新颖的centerLoss,该方法在2016年的ECCV上提出：A Discriminative Feature Learning Approach for Deep Face Recognition \
  代码链接:http://ydwen.github.io/papers/WenECCV16.pdf  \
  论文链接：https://github.com/pangyupo/mxnet_center_loss

  作者认为，在使用传统的softmaxLoss时，该loss仅仅能够学习到如何将不同的类别进行区分的特征，该特征应用于人脸方面是不具有很强的判别性的，特别是针对开放集中非训练集样本的预测问题,如图1。同时，作者大致介绍了contrastive个tripletLoss这两种具有很好能够训练出良好特征的loss，由于要构造训练对，并且训练时候收敛速度较慢 ![](https://github.com/zhanglaplace/caffe_newLayer/blob/master/imgs/centerLoss_F1.png)


  >本文作者： 张峰
  >本文链接： https://zhanglaplace.github.io/2017/10/19/Caffe_Net/
  >版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！

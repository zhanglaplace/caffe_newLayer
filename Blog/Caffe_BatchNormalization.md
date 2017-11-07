---
title: Caffe Batch Normalization推导
date: 2017-11-06 13:44:33
tags: [Caffe,DeepLearning]
categories: Caffe
---

# Caffe BatchNormalization 推导
  总所周知，BatchNormalization通过对数据分布进行归一化处理，从而使得网络的训练能够快速并简单，在一定程度上还能防止网络的过拟合，通过仔细看过Caffe的源码实现后发现，Caffe是通过BN层和Scale层来完整的实现整个过程的。
<!--more-->
  那么再开始前，先进行必要的公式说明：定义$L$为网络的损失函数，BN层的输出为$y$，根据反向传播目前已知 $\frac{\partial L}{\partial y_i}$,其中：
   $$y_i = \frac{x_i-\overline{x}}{\sqrt{\delta^2+\epsilon}},\quad\overline x = \frac{1}{m}\sum_{i=1}^{m}x_i,\quad \delta^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i-\overline x)^2,\quad 求\frac{\partial L}{\partial x_i}$$

  推导的过程中应用了链式法则：
  $$ \frac{\partial L}{\partial x_i} = \sum_{j=1}^{m}{\frac{\partial L}{\partial y_j}\*\frac{\partial y_j}{\partial x_i}} $$
  则只需要着重讨论公式 $\frac{\partial y_j}{\partial x_i}$

  分布探讨：

  (1) $\overline x$对$x_i$的导函数
  $$\frac{\partial \overline x}{\partial x_i} = \frac{1}{m} $$

  (2) $\delta^2$对$x_i$的导函数
  $$\frac{\partial \delta^2}{\partial x_i} = \frac{1}{m}(\sum_{j=1}^{m}2\*(x_j-\overline x)\*(-\frac{1}{m}))+2(x_i-\overline x)$$
  由于 $\sum_{j=1}^{m}2\*(x_j-\overline x) = 2\* \sum_{i=1}^{m}x_i - n\*\overline x = 0$

  所以： $\frac{\partial \delta^2}{\partial x_i} = \frac{2}{m}\*(x_i-\overline x)$


  具体推导：
  $$\frac{\partial y_j}{\partial x_i} = \frac{\partial{\frac{x_j -\overline x}{\sqrt{\delta^2+\epsilon}}}}{\partial x_i} $$
  此处当$j$等于$i$成立时时，分子求导多一个 $x_i$的导数

  $$\frac{\partial y_j}{\partial x_i} = -\frac{1}{m}(\delta^2+\epsilon)^{-1/2}-\frac{1}{m}(\delta^2+\epsilon)^{-3/2}(x_i-\overline x)(x_j - \overline x)\quad\quad i \neq j $$
  $$\frac{\partial y_j}{\partial x_i} = (1-\frac{1}{m})(\delta^2+\epsilon)^{-1/2}-\frac{1}{m}(\delta^2+\epsilon)^{-3/2}(x_i-\overline x)(x_j - \overline x)\quad\quad i = j$$

  根据上式子，我们代入链式法则的式子
  $$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y_i}\*(\delta^2+\epsilon)^{-1/2} + \sum_{j=1}^{m}\frac{\partial L}{\partial y_j}\*(-\frac{1}{m}(\delta^2+\epsilon)^{-1/2}-\frac{1}{m}(\delta^2+\epsilon)^{-3/2}(x_i-\overline x)(x_j-\overline x))$$

  我们提出 $(\delta^2+\epsilon)^{-1/2}:$
  $$\frac{\partial L}{\partial x_i} = (\delta^2+\epsilon)^{-1/2}(\frac{\partial L}{\partial y_i}- \sum_{j=1}^{m}\frac{\partial L}{\partial y_j}\frac{1}{m}-\sum_{j=1}^{m}\frac{\partial L}{\partial y_j}\frac{1}{m}(\delta^2+\epsilon)^{-1}(x_i-\overline x)(x_j-\overline x))
  \\
  =(\delta^2+\epsilon)^{-1/2}(\frac{\partial L}{\partial y_i}- \sum_{j=1}^{m}\frac{\partial L}{\partial y_j}\frac{1}{m}-\sum_{j=1}^{m}\frac{\partial L}{\partial y_j}\frac{1}{m}y_jy_i   \\
  =(\delta^2+\epsilon)^{-1/2}(\frac{\partial L}{\partial y_i}- \frac{1}{m}\sum_{j=1}^{m}\frac{\partial L}{\partial y_j}-\frac{1}{m}y_i\sum_{j=1}^{m}\frac{\partial L}{\partial y_j}y_j)$$

  至此，我们可以对应到caffe的具体实现部分
```cpp
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
 //
 // dE(Y)/dX =
 //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
 //     ./ sqrt(var(X) + eps)
 //
 // where \cdot and ./ are hadamard product and elementwise division,
 ```

 >本文作者： 张峰
 >本文链接：[http://www.enjoyai.site/2017/11/06/](http://www.enjoyai.site/2017/11/06/Caffe_BatchNormalization/ )
 >版权声明：本博客所有文章，均采用CC BY-NC-SA 3.0 许可协议。转载请注明出处！

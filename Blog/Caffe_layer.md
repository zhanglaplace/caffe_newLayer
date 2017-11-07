---
title: Caffe Layer分析
date: 2017-10-19 11:31:00
tags: [Caffe,DeepLearning]
categories: Caffe
---
![](http://images2017.cnblogs.com/blog/888534/201710/888534-20171019224727334-359802148.png)

# Caffe_Layers

### 1.基本数据结构
```cpp
    //Layer层主要的的参数
    LayerParamter layer_param_; // protobuf内的layer参数
    vector<shared_ptr<Blob<Dtype>*>>blobs_;//存储layer的参数，
    vector<bool>param_propagate_down_;//表示是否计算各个blobs反向误差。

```
<!--more-->
### 2.主要函数接口
``` cpp
    virtual void SetUp(const vector<Blob<Dtype>*>&bottom,
                        vector<Blob<Dtype>*>& top);
    Dtype Forward(const vector<Blob<Dtype>*>&bottom,
                        vector<Blob<Dtype>*>&top);
    void Backward(const vector<Blob<Dtype>*>&top,
    const vector<bool>param_propagate_down,vector<Blob<Dtype>*>& bottom);
```

### 3.具体的Layer分析
    具体的常用Layer分析
#### (1) 数据层(DataLayer)
数据通过数据层进入Layer,可以来自于数据库(LevelDB或者LMDB),也可以来自内存，HDF5等
```cpp
    //Database：类型 Database
    //必须参数 source,batch_size
    //可选参数：rand_skip,mirror,backend[default LEVELDB]

    // In-Memory：类型 MemoryData
    // 必选参数：batch_size，channels,height,width

    //HDF5 Input:类型 HDF5Data
    //必选参数: source,batch_size

    //Images : 类型 ImageData
    //必要参数：source(文件名label),batch_size
    //可选参数：rand_skip,shuffle,new_width,new_height;

```
#### (2) 激励层(neuron_layers)
    一般来说，激励层是element-wise，输入输出大小相同，一般非线性函数
    输入：n\*c\*h\*w
    输出：n\*c\*h\*w

```cpp
    //ReLU/PReLU
    //可选参数 negative_slope 指定输入值小于零时的输出。
    // f(x) = x*(x>0)+negative_slope*(x<=0)
    //ReLU目前使用最为广泛，收敛快，解决梯度弥散问题
    layer{
        name:"relu"
        type:"ReLU"
        bottom:"conv1"
        top:"conv1"
    }

    //Sigmoid
    //f(x) = 1./(1+exp(-x)); 负无穷--正无穷映射到-1---1
    layer{
        name:"sigmoid-test"
        bottom:"conv1"
        top:"conv1"
        type:"Sigmoid"
    }

```
#### (3) 视觉层(vision_layer)
    常用layer操作
```cpp
     //卷积层(Convolution):类型Convolution
     //包含学习率，输出卷积核，卷积核size，初始方式，权值衰减
     //假使输入n*ci*hi*wi,则输出
     // new_h = ((hi-kernel_h)+2*pad_h)/stride+1;
     // new_w = ((wi-kernel_w)+2*pad_w)/stride+1;
     //输出n*num_output*new_h*new_w;
     layer{
         name: "conv1"
         type: "CONVOLUTION"
         bottom: "data"
         top: "conv1"
         blobs_lr: 1
         blobs_lr: 2
         weight_decay: 1
         weight_decay: 0
         convolution_param {
             num_output: 96
             kernel_size: 11
             stride: 4
             weight_filler {
                 type: "gaussian"
                std: 0.01
             }
             bias_filler {
               type: "constant"
               value: 0
             }
          }
     }

    //池化层(Pooling) 类型 Pooling
    // (hi-kernel_h)/2+1;
    layer{
        name:"pool1"
        type:"POOLING"
        bottom:"conv1"
        top:"conv1"
        pooling_param{
            pool:MAX //AVE,STOCHASTIC
            kernel_size:3
            stride:2
        }
    }

    //BatchNormalization
    // x' = (x-u)/δ ;y = α*x'+β;

```

#### (4) 损失层(Loss_layer)
    最小化输出于目标的LOSS来驱动学习更新

```cpp
    //Softmax

```

### 4.说明
SetUp函数需要根据实际的参数设置进行实现，对各种类型的参数初始化；Forward和Backward对应前向计算和反向更新，输入统一都是bottom，输出为top，其中Backward里面有个propagate_down参数，用来表示该Layer是否反向传播参数。
在Forward和Backward的具体实现里，会根据Caffe::mode()进行对应的操作，即使用cpu或者gpu进行计算，两个都实现了对应的接口Forward_cpu、Forward_gpu和Backward_cpu、Backward_gpu，这些接口都是virtual，具体还是要根据layer的类型进行对应的计算（注意：有些layer并没有GPU计算的实现，所以封装时加入了CPU的计算作为后备）。另外，还实现了ToProto的接口，将Layer的参数写入到protocol buffer文件中。

>本文作者： 张峰
>本文链接： https://zhanglaplace.github.io/2017/10/19/Caffe_layer/
>版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！

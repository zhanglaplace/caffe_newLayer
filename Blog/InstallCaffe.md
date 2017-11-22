---
title: Caffe 安装分析
date: 2017-11-21 15:40:10
tags: [Caffe,DeepLearning]
categories: Caffe
---


# Caffe安装
  实际上在windows上安装过多次caffe了，无论是BLVC版本的还是Microsoft版本的，ubuntu的按照也进行过，这段时间在自己笔记本上
又折腾了下caffe安装，发现其实直接照着官方的是最方便快捷的。
  具体可以参照 ![Installation_instructions](http://caffe.berkeleyvision.org/install_apt.html)
# 具体步骤
  根据系统的不同，ubuntu分为两种安装方式。Ubuntu17.04以即上的可以直接apt-get
```shell
  sudo apt install caffe-cpu # cpu only
  sudo apt install caffe-cuda # gpu
```

  其他版本的ubuntu也可以分为两种方式安装，但是依赖项是必须的，本文仅介绍简易的软件源中快速安装。源码安装可以参考本人的另外一篇博客：![ubunt16.04 cud8.0 caffe 安装](http://www.cnblogs.com/LaplaceAkuir/p/6262632.html)

## Nvidia显卡驱动
  由于要使用GPU，所以先要查看自己显卡所匹配的显卡驱动，网址：![nvidia](http://www.nvidia.com/Download/index.aspx?lang=en-us) ,下载run文件。
  由于目前显卡和cuda更新迅速，容易造成笔记本循环登录，因此安装显卡驱动是关闭图形界面。
```shell
  # ctrl +alt +F1 进入tty1，
  sudo service lightdm stop
  sudo ./Nvidia-.....run 执行安装
  sudo reboot
```

## Cuda和CuDnn
  安装较为简单，官网下载，在安装cuda是需要注意显卡安装选项选择no即可
  
```shell
    sudo sh cuda_8.0.44_linux.run --override
    # 安装结束后
    sudo vim ~/.bashrc  //末尾添加
    export CUDA_HOME=/usr/local/cuda-8.0
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-8.0/bin:$PATH
    source ~/.bashrc

    # 测试
    cd /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery
    make -j32
    sudo ./deviceQuery
```
   cudnn下载后接下的include lib拷贝到cuda的安装路径，并设置链接。
```shell
   sudo ln -s libcudnn.so.xxx libcudnn
   sudo ln -s libcudnn.so.xx libcudd.so
   sudo ldconfig
 ```
# 其他依赖项
  其他依赖项安装可以直接从软件源获取，当然也可以自己源码安装。
```shell
  # protobuf,snappy,leveldb,opencv,hdf5,boost ,python-opencv,glog ,gflag,lmdb
  sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
  sudo apt-get install --no-install-recommends libboost-all-dev
  sudo apt-get install python-dev python-opencv
  sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```
  关于blas可以选择atlas，openblas和MKL，由于后续cmake方式安装默认atlas，所以本人也用次
```shell
  sudo apt-get install libatlas-base-dev
  # openblas也很方便
  sudo apt-get install libopenblas-dev
```
  Matlab的接口可以自己先安装matlab ，此处省略，同时python可以安装anconda来管理库

# 安装

## Caffe
  下载BVLC的caffe
```shell
  git clone --recursive https://github.com/BVLC/caffe
```
## 编译
### 1.Make方式
  Make方式为官方的编译方式，但是在后续caffe的使用时会稍显麻烦，这里要注意根据安装的库以及自己是否使用gpu、cudnn以及bals的选择等作出修改
```shell
  cp Makefile.config.example Makefile.config
  # For CPU & GPU accelerated Caffe, no changes are needed.
  #For cuDNN acceleration using NVIDIA’s proprietary cuDNN software, uncomment the USE_CUDNN := 1 switch in #Makefile.config. cuDNN is sometimes but not always faster than Caffe’s GPU acceleration.
  #For CPU-only Caffe, uncomment CPU_ONLY := 1 in Makefile.config.
  # Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)
  make all -j8
  make test
  make runtest

```
### 2.Cmake方式
  Cmake方式针对自己使用Caffe以及从软件源安装Caffe的用户来说简直不要更方便.
```shell
  mkdir build
  cd build
  cmake ..
  make all
  make install
  make runtest
```
  由于自己使用Caffe不仅仅是停留在训练，可能很多都要具体的测试实际的项目，因此相比于Make方式，Cmake的优势就大大体现出来了。具体例子可以在我的github上看到![https://github.com/zhanglaplace/MTCNN-Accelerate-Onet](https://github.com/zhanglaplace/MTCNN-Accelerate-Onet)
  编译自己的项目，仅仅需要写一个简单的CMakeLists.txt文件，并且文件内的内容可以保证百分之九十的不变，这使得验证算法和项目变得相当方便.(强烈推荐)
```cpp
  cmake_minimum_required(VERSION 2.9)
  project(MTCNN_Accelerate-Onet)  // 根据自己工程名字修改

  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  set(CMAKE_CXX_STANDARD 11)

  find_package(OpenCV)

  find_package(Caffe REQUIRED)
  #message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
  include_directories(${Caffe_INCLUDE_DIRS})

  set(SOURCE_FILES main.cpp mtcnn.cpp mtcnn.h) // 根据自己实际源码修改
  add_executable(MTCNN_Accelerate-Onet ${SOURCE_FILES})

  target_link_libraries(MTCNN_Accelerate-Onet ${OpenCV_LIBS} )
  target_link_libraries(MTCNN_Accelerate-Onet ${Caffe_LIBRARIES})

```

>本文作者： 张峰
>本文链接： https://zhanglaplace.github.io/2017/10/19/Caffe_Net/
>版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！

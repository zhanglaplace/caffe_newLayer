---
title: Caffe DataLayer分析
date: 2017-10-20 11:16:01
tags: [Caffe,DeepLearning]
categories: Caffe
---

### Caffe DataLayer分析
  Caffe的$data$层作为网络的起始部分，是网络的最底层，其不仅提供了数据的输入，也提供了数据的格式转换，而数据的来源可以来自于高效率的数据库 $(lmdb,levelDb)$,也可以来自于内存，$HDF5$文件等。

#### $（1）\,DataLayer$
  $Data$作为最常用的$caffe$网络训练的数据输入，其参数也比较多，具体可以参见$proto$中关于$DataParameter$的定义,通过$Datalayer$可以完成数据的转化，需要指明$source,batch\_size$,也可以指明$mirror,crop,backend$等

1.基本数据成员
```cpp
    DataRead reader_;//data_reader.hpp,cpp用以定义读取数据的操作
```
2.基本成员函数
  主要包括$Layersetup$函数，该函数为整个网络架构搭建的开始
```cpp
    //LayerSetup
    template <typename Dtype>
    void DataLayer<Dtype>::LayerSetup(const vector<Blob>*>& bottom,
      const vector<Blob>*>& top){
      const DataParamter& param = this->layer_param_->data_param();
      const int batch_size = param.batch_size();
      Datum& datum = *(reader_.full().peek()); //数据库的读取操作




    }

```


>本文作者： 张峰
>本文链接：[https://zhanglaplace.github.io/2017/10/20]( https://zhanglaplace.github.io/2017/10/20/Caffe%20Loss%E5%88%86%E6%9E%90/)
>版权声明：本博客所有文章，均采用CC BY-NC-SA 3.0 许可协议。转载请注明出处！

---
title: Caffe Blob分析
date: 2017-10-18 00:03:10
tags: [Caffe,DeepLearning]
categories: Caffe
---

# Caffe_blob

### 1.基本数据结构

  Blob为模板类，可以理解为四维数组，n \* c \* h \* w的结构,Layer内为blob输入data和diff，Layer间的blob为学习的参数.内部封装了SyncedMemory类,该类负责存储分配和主机与设备的同步
```cpp
protected:
    shared_ptr<SyncedMemory> data_; // data指针
    shared_ptr<SyncedMemory> diff_; // diff指针
    vector<int> shape_; // blob形状
    int count_; // blob的nchw
    // 当前的Blob容量，当Blob reshape后count> capacity_时，capacity_ = count_;
    // 重新new 然后 reset data和 diff
    int capacity_;

```
<!--more-->
### 2.常用函数
    Blob类中常用的函数如下所示
```cpp
    Blob<float>test;
    //explicit关键字的作用是禁止单参数构造函数的隐式转换
    explicit Blob(const int num, const int channels, const int height,
      const int width);
    test.shape_string();//初始为空 0 0 0 0
    //Reshape函数将num,channels,height,width传递给vector shape_
    test.Reshape(1,2,3,4);// shape_string() 1,2,3,4

    test.shape(i);// NCHW
    test.count(int start_axis,int end_axis); // start_axis---end_axis .x* shape[i]
    test.count();// nchw  count(1) chw count(2) hw.....
    //shared_ptr<SyncedMemory> data_->cpu_data();
    const float* data = test.cpu_data();
    const float* diff = test.cpu_diff();
    float* data_1 = test.mutable_cpu_data();//mutable修饰的表示可以修改内部值
    float* diff_1 = test.mutable_cpu_diff();
    test.asum_data();//求和 L1范数
    test.sumsq_data();//平方和 L2范数
    test.Update();//data = data-diff;
    a.ToProto(BlobProto& bp,true/false);//(FromProto)
    // if < 0 ,return num_axis()+axis_index;//索引序列
    int index = a.CanonicalAxisIndex(int axis_index);
    int offset(n,c,h,w);//((n*channels()+c)*height()+h)*width()+w
    float data_at(n,c,h,w);//return cpu_data()[offset(n,c,h,w)];
    float diff_at(n,c,h,w);//return cpu_diff()[offset(n,c,h,w)];
    inline const shared_ptr<SyncedMemory>& data() const{return _data};
    void scale_data(Dtype scale_factor);// data乘以一个标量。同理 scale_diff();
    void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);  // copy_diff是否复制diff
```

### 3.写入磁盘操作

```cpp
  //Blob内部值写入到磁盘
  Blob<float>a;
  a.Reshape(1,2,3,4);
  const int count = a.count();
  for (size_t i = 0; i < count; i++) {
      a[i] = i;//init the test Blob
  }
  BlobProto bp,bp2;
  a.ToProto(&bp,true);//写入data和diff到bp中
  WriteProtoToBinaryFile(bp,"a.blob");//写入磁盘
  ReadProtoFromBinaryFile("a.blob",&bp2);//从磁盘读取blob
  Blob<float>b;
  b.FromProto(bp2,true);//序列化对象bp2中克隆b，完整克隆
  for (size_t n = 0; n < b.num(); n++) {
      for (size_t c = 0; c < b.channels(); c++) {
         for (size_t h = 0; h < b.height(); h++) {
             for (size_t w = 0; w < b.width(); w++) {
                 cout<<"b["<<n<<"]["<<c<<"]["<<h<<"]["<<w<<"]["<<w<<"]="<<
                 b[(((n*b.channels()+c)*b.height)+h)*b.width()+w]<<endl;
                 //(((n*c+ci)*h+hi)*w+wi)
             }
         }
      }
  }

```

### 4.部分函数的具体实现
    本部分的实现未考虑参数是否合理。一般操作blob需要分CPU和GPU,采用math_functions具体计算
```cpp
  template <typename Dtype>
  void Blob<Dtype>::Reshape(const vector<int>& shape){//reshape操作
    count_ = 1;//初始count_ NCHW;
    shape_.resize(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
      count_ *= shape[i];
      shape_[i] = shape[i];
      if (count_ > capacity_) { //reshape的size大于了目前的最大容量
         capacity_ = count_;
         data_.reset(new SyncedMemory(capacity_*sizeof(Dtype)));
         diff_.reset(new SyncedMemory(capacity_*sizeof(Dtype)));
      }
    }
  }

 template <typename Dtype>
 void Blob<Dtype>::Reshape(int n,int c,int h ,int w){//reshape操作
   vector<int>shape(4);
   shape[0] = n;
   shape[1] = c;
   shape[2] = h;
   shape[3] = w;
   Reshape(shape);
 }

 template <typename Dtype>
 const Dtype* Blob<Dtype>::cpu_data(){
   //实际调用的shared_ptr<SyncedMemory>data_->cpu_data();,同理cpu_diff();
   CHECK(data_);
   return (const Dtype*)data_->cpu_data();
 }

template <typename Dtype>
void Blob<Dtype>::Updata(){ //data = data-diff;需要判断cpu OR gpu
  switch (data_->head()) {
    case SyncedMemory::HEAD_AT_CPU:
      caffe_axpy<Dtype>(count_,Dtype(-1),
      static_cast<const<Dtype*>(diff_->cpu_data()),
      static_cast<Dtype*>(data_->mutable_cpu_data()));
  }
    case SyncedMemory::HEAD_AT_GPU://在gpu或者CPU/GPU已经同步
    case SyncedMemory::SYNCED:
    #ifndef CPU_ONLY
      caffe_gpu_axpy<Dtype>(count_.Dtype(-1),
      static_cast<const<Dtype*>(diff_->gpu_data()),
      static_cast<Dtype*>(data_->mutable_gpu_data()))
}

template <typename Dtype> //从source 拷贝数据,copy_diff控制是拷贝diff还是data
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {  //copy diff
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}


template <typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto,bool write_diff){
    proto->clear_shape();
    for (size_t i = 0; i < shaoe_.size(); i++) {
        proto->mutable_shape()->add_dim(shape_[i]);
    }
    proto->clear_data();
    proto->clear_diff();
    const Dtype* data_vec = cpu_data();
    for (size_t i = 0; i < count_; i++) {
        proto->add_data(data_vec[i]);//data写入proto
    }
    if (write_diff) {
        const Dtype* diff_vec = cpu_diff();
        for (size_t i = 0; i < count_; i++) {
            proto->add_diff(diff_vec[i]);//diff写入proto
        }
    }
}

```

### 5.说明
```cpp
    /*Blob作为一个最基础的类，其中构造函数开辟一个内存空间来存储数据，Reshape
    函数在Layer中的reshape或者forward操作中来调整top的输出维度。同时在改变Blob
    大小时， 内存将会被重新分配如果内存大小不够了，并且额外的内存将不会被释放。
    对input的blob进行reshape, 若立马调用Net::Backward是会出错的，因为reshape
    之后，要么Net::forward或者Net::Reshape就会被调用来将新的input shape传播
    到高层 */

```
>本文作者： 张峰
>本文链接： https://zhanglaplace.github.io/2017/10/18/Caffe_blob/  
>版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！

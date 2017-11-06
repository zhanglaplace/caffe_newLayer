---
title: Caffe CommonLayer分析
date: 2017-10-20 11:16:01
tags: [Caffe,DeepLearning]
categories: Caffe
---

### Caffe CommonLayer分析
  $Caffe$中包含了很多通用的功能层，包含了$concat$,$slice$,$split$,$crop$,$flip$,$scale\_layer$等,这些层在网络中经常被使用，本文也将对其中的常见layer进行说明与源码分析。

#### 1.常用$Layer$

##### (1) $CropLayer$
  CropLayer完成数据的裁剪，输入两个 $bottom,bottom[0]$ 为原始数据，$bottom[1]$ 为裁剪后
的输出尺寸，输出 $top[0]$ 为裁剪后的数据，尺寸与 $bottom[1]$ 相同，其中有$axis 控制裁剪
的起始轴,offset表示对应裁剪轴的起始位置。举例说明：
$$bottom[0]的shape:[32,64,512,512],bottom[1]的shape:[32,32,256,256] \\
axis = 1 \qquad offset = [:,16,128,128] \\
则:top[1]的为bottom[0][:,16+bottom[1].shape(1),128+bottom[1].shape(2),128+bottom[1].shape(3)))$$
下面会进行具体的代码解释说明

1.基本成员变量
 基本成员变量，记录开始的axis和每个shape的起始偏移
 ```cpp
  vector<int>offsets_;
  int axis;
```

2.基本成员函数
  基本成员函数包括LayerSetup,Reshape,forward,Backward,crop_copy，具体实现如下
```cpp
  //LayerSetup 主要完成proto的参数提取过程
  template <typename Dtype>
  void CropLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    const CropParameter& crop_param = this->layer_param_.crop_param();
    CHECK_EQ(bottom.size(),2);//必须是2个
    int input_dim = bottom[0]->num_axes();// 一般为4， 即shape_.size()
    const int start_axis = bottom[0]->CanonicalAxisIndex(crop_param.axis());
    // 这里的axis要判断是否小于Input_dim
    if (crop_param.offset_size() > 1) { // offset_size() == offset.size()
        CHECK_EQ(start_axis+crop_param.offset_size(),input_dim);
        //保证起始后的axis均有offset
    }
  }

  // Reshape,确定offsets和crop_size的尺寸
  template <typename Dtype>
  void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
      const CropParameter param = this->layer_param_.crop_param();
      const int dim = bottom[0]->num_axes();
      const int start_axis = param.axis();
      offsets_ = std::vector<<int>(input_dim,0);
      vector<int> new_shape(bottom[0]->shape());
      for (size_t i = 0; i < dim; i++) {
          int crop_offset = 0; //偏移量
          int new_size = bottom[0]->shape(i);//每个shape的size
          // i >= start_axis的时候才crop,否则不改变shape
          if ( i >= start_axis) {
             new_size = bottom[1].shape(i);
             if (param.offset_size() == 1) {//如果只给出一个offset默认都一样
                crop_offset = param.offset(0);
             }
             else if(param.offset_size() > 1){
               crop_offset = param.offset(i-start_axis);
             }
             CHECK_GE(bottom[0]->shape(i),crop_offset+bottom[1]->shape(i));
          }
          new_shape[i] = new_size;
          offsets_[i] = crop_offset;
      }
      top[0]->Reshape(new_shape);
    }
```
  $Forward$ 的前向过程设计到元素复制的问题，使用 $crop\_copy$ 函数单独实现，
```cpp
  template <typename Dtype>
  void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
    const vector<Blob<Dtype>* >& top){
    vector<int>indices(top[0].num_axes(),0);
    const Dtype* bottom_data = bottom[0]->cpu_data();//输入
    Dtype* top_data = top[0]->mutable_cpu_data();//输出
    crop_copy(botoom,top,offsets,indices,0,botton_data,top_data,true);
  }

  template <typename Dtype>
  void CropLayer<Dtype>::crop_copy(const vector<Blob<Dtype>*>& bottom,
    const vecotr<Blob<Dtype>*>& top,const vector<int>& offsets,
    vector<int>& indices,int cur_dim,const Dtype* src_data,
    const Dtype* dst_data,bool is_forward){

    //循环赋值每个维度
    for (size_t i = 0; i < top[0]->shape(cur_dim); i++) {

    }
  }
```


##### (2) $AccracyLayer$
  Accracy_layer用以统计训练过程中样本预测的准确率，根据label值与top_K的得分标签的对比，来计算准确率，因此可以在prototxt中设置top_k参数，观察训练状况。
1.基本数据成员
```cpp
  int label_axis_;//实际上就是第一个channels是label
  int outer_num_;//BATCH_SIZE
  int inner_num_;//一般为 1 即H*W
  int top_k;
  bool has_ignore_label;
  int ignore_label;
  Blob<Dtype>nums_buffer_;//统计每个类别的样本数量
```
2.基本成员函数
  基本成员函数包括$LayerSetup$,$Reshape$,$forward$,其中参数的设置和读取发生在$LayerSetup$和$Reshape$上,acuracy可以显示训练中的信息，稍加修改也可以显示$Recall,F1$值等信息,同时
  类别较少的时候，加入一个输出$top$，即可显示出每个类别的训练中的$accuracy$情况，具体实现如下:
  ```cpp
    //layersetup 仅仅完成参数的读取
    template <typename Dtype>
    void AccuracyLayer<Dtype>::LayerSetup(const vector<Blob>*>& bottom,
      const vector<Blob>*>& top){
      const AccuracyParameter& param = this->layer_param_.accuracy_param();
      top_k = param.top_k();
      label_axis_ = bottom[0]->CanonicalAxisIndex(param.axis());;
      has_ignore_label = param.has_ignore_label();
      if (has_ignore_label) {
         ignore_label = param.ignore_label();
      }
    }

    //Reshape
    // 多个top的时候，第一个top为整体的AC，第二个top为每个类别的ac
    template <typename Dtype>
    void AccuracyLayer<Dtype>::Reshape(const vector<Blob>*>& bottom,
      const vector<Blob>*>& top){
      outer_num_ = bottom[0]->count(0,label_axis_);//N
      inner_num_ = bottom[0]->count(label_axis_+1);//1*1 (H*W)
      vector<int>top_shape(0);
      top[0]->Reshape(top_shape);
      if (top.size() > 1) {
         //每个类别是一个向量，每个类别都需要统计单独的accuracy，而不是整体的
         vector<int>top_shape_pre_class(1);
         top_shape_pre_class[0] = bottom[0]->shape(label_axis_);//N个类别
         top[1]->Reshape(top_shape_pre_class);
         nums_buffer_.Reshape(top_shape_pre_class);
      }
    }


    //Forward_cpu,top[1]为C*1 ,Top[0]为1*1*1*1
    //前向过程，如果多个top则需要统计每个类别的accuracy保存到top[1]中
    template <typename Dtype>
    void AccracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>* >& top){
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* label = bottom[1]->cpu_data();
      const int dim = bottom[0]->count()/outer_num_;//类别数目
      if (top.size() > 1) {
         caffe_set(nums_buffer_.count(),0,nums_buffer_.mutable_cpu_data());
         caffe_set(top[1]->count(),0,top[1]->mutable_cpu_data());
      }
      Dtype accuracy = 0;
      int count = 0;
      for (size_t i = 0; i < outer_num_; i++) { //N
        for (size_t j = 0; j < inner_num_; j++) { //1*1
          const int label_value = static_cast<int>(label[i*inner_num_+j]);
          if (has_ignore_label && ignore_label == label_value) {
              continue;//当前label是忽略的label
          }
          if (top->size() > 1) {
            nums_buffer_.mutable_cpu_data()[label_value]++;//类别数目+1
          }
          //看top_k的最大
          std::vector<std::pair<Dtype,int>> bottom_data_vector;
          for (size_t k = 0; k < dim ; k++) {
              bottom_data_vector.push_back(
                std::make_pair(bottom_data[i*dim+k*inner_num_+j]));
          }
          //最大堆排序
          std::partial_sort(
          bottom_data_vector.begin(),bottom_data_vector.begin()+top_k,
          bottom_data_vector.end(),std::greater<std::pair<Dtype, int>>());

          //查找top_k有没有真实的label
          for (size_t i = 0; i < top_k; i++) {
            if (bottom_data_vector[i].second == label_value) {
              ++accuracy;
              if (top.size() > 1) {
                //每类样本预测正确的数目+1
                top[1]->mutable_cpu_data()[label_value]++;
              }
              break;
            }
          }
          count++;
        }
      }
      //全部mini的样本循环完成后
      top[0]->mutable_cpu_data()[0] = accuracy/count;
      if (top.size() > 1) {
        for (size_t i = 0; i < dim; i++) {//dim表示类别
            top[1]->mutable_cpu_data()[i] =
            nums_buffer_.cpu_data()[i] == 0 ? 0;
            top[1]->cpu_data()[i]/nums_buffer_.cpu_data()[i];
        }
      }
    }
```

##### (3) $EltwiseLayer$
  $EltwiseLayer$在深度网络中运用非常广泛，常用与多$layer$的合并，在$ResidualNet$中用以连接$block$与$x$部分，其组合方式有$prod,sum,max$,最常见的为$sum$和$max$,由于组合的方式有多种，因此在进行前向和后向的分析的时候需要按照多种情况进行分析，详细的代码解析如下所示：
1.基本数据成员
```cpp
    EltwiseParameter_EltwiseOp op_;// sum,prod,max 实际是个enum数据
    vector<Dtype> coeffs_; // 代表操作参数 如果-1，代表a-b
    Blob<int> max_idx;
    bool stable_prod_grad_;//只针对PROD，点乘模式
```
2.基本成员函数
```cpp

    //LayerSetup 完成参数的读取
    template <typename Dtype>
    void EltwiseLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
      const EltwiseParameter& param = this->layer_param_.eltwise_param();
      op_ = param.operation();
      coeffs_ = vector<Dtype>(bottom.size(),1);
      if (param.coeff_size()) {//每个layer的前面的标量
        for (size_t i = 0; i < param.coeff_size(); i++) {
          coeffs_[i] = param.coeff(i); //1 -1等参数
        }
      }
      stable_prod_grad_ = param.stable_prod_grad();
    }

    //Reshape过程，完成topshape的构造
    template <typename Dtype>
    void EltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
      //bottom的shape要完全一样
      for (size_t i = 1; i < bottom.size(); i++) {
          CHECK_EQ(bottom[i]->shape() == bottom[0]->shape())
      }
      top[0]->ReshapeLike(*bottom[0]);//当然输出也要一样
      if (this_->layer_param_.eltwise_param().operation()==
          EltwiseParameter_EltwiseOp_Max && top.size() == 1) {
          max_idx_.Reshape(bottom[0]->shape());//记录每个的maxid
      }
    }

  //Forward_cpu,完成layer的前向操作
  template <typename Dtype>
  void EltwiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    //临时变量,用以MAX操作
    const Dtype* bottom_data = NULL;
    const int count = top[0]->count();//
    Dtype* top_data = top[0]->mutable_cpu_data();
    switch (op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      caffe_mul(count,bottom[0]->cpu_data(),bottom[1]->cpu_data(),top_data);
      for (size_t i = 2; i < bottom.size(); i++) {
        caffe_mul(count,top_data,bottom[i]->cpu_data(),top_data);
      }
      break;
    case  EltwiseParameter_EltwiseOp_SUM:
      caffe_set(count,Dtype(0),top_data);//先初始输出为0
      for (size_t i = 0; i < bottom.size(); i++) {
        caffe_axpy(count,coeffs_[i],bottom[i]->cpu_data(),top_data);
      }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      caffe_set(count,-1,max_idx_.mutable_cpu_data());
      caffe_set(count,Dtype(-FLT_MIN),top_data);
      for (size_t i = 0; i < bottom.size(); i++) {
        bottom_data = bottom[i]->cpu_data();
        for (size_t j = 0; j < count; j++) { //整体遍历
           if (bottom_data[j] > top_data[j]) {
              top_data[j] = bottom_data[j];
              max_idx_.mutable_cpu_data()[j] = i;
           }
         }
      }
    default:
      //Not Support;
    }
  }

  //Backward_cpu,完成layer的反向操作
  // 当method == SUM的时候，bottom_diff[i] = coeffs_[i]* top_diff;
  // 当method == product的时候,bottom_diff[i] = top_diff*top_data/bottom_data[i]
  // 当method == Max 的时候，bottom_diff[i] = top_diff*(j==max_idx_?1:0)
  template <typename Dtype>
  void EltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* top_data = top[0]->cpu_data();
      const int cont = top[0]->count();
      for (size_t i = 0; i < bottom.size(); i++) {
        if (propagate_down[i]) { //需要backward才考虑
            const Dtype* bottom_data = bottom[i]->cpu_data();
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
            switch (op_) {
              //bottom_diff[i] = top_diff*top_data/bottom_data[i]
              case EltwiseParameter_EltwiseOp_PROD://点成操作
                if (stable_prod_grad_) { //渐进梯度的实现top_data/bottom[i]
                  bool initiaized = false;
                  for (size_t j = 0; j < bottom.size(); j++) {
                    if(i == j) continue; //top/bottom[i] == bottom[j]连乘
                    if (!initiaized) { //初始化
                      //用bottom[j]初始一下bottom_diff
                      caffe_copy(count,bottom[j]->cpu_data(),bottom_diff);
                      initiaized = true;
                    }
                    else{
                      caffe_mul(count,bottom[j]->cpu_dat(),bottom_diff,
                              bottom_diff);
                    }
                  }
                }
                else{
                  caffe_div(count,top_data,bottom_data,bottom_diff);
                }
                caffe_mul(count,bottom_diff,top_diff,bottom_diff);
                break;

              //bottom_diff[i] = coeffs_[i]* top_diff;
              case  EltwiseParameter_EltwiseOp_SUM://sum操作
                  if (coeffs_[i] == Dtype(1)) {
                    caffe_copy(count,top_diff,bottom_diff);
                  }
                  else{
                    caffe_scale(count,coeffs_[i],top_diff,bottom_diff);
                  }
                  break;

              //bottom_diff[i] = top_diff*(j==max_idx_?1:0)
              case EltwiseParameter_EltwiseOp_MAX: //max操作
                  for (size_t j = 0; j < count; j++) {
                     if (max_idx_.cpu_data()[j] == i) {
                       bottom_diff[j] = top_diff[j];
                     }
                     else{
                       bottom_diff[j] = Dtype(0);
                     }
                  }
                  break;

              default:
              //  Not Support

            }
        }
      }
  }

```
$Eltwise$的$backward$的$product$模式有两种实现方式:
(1) top_data/bottom[i];
(2) $\prod_{j=0,j!=i}^{n-1}bottom[i]$

##### (4)$ConcatLayer$
  同$Eltwise$类似，$Concat$在多特征图的融合方面使用也极为广泛(denseNet,Dpn),$concat$中有$axis$和$concat\_dim$控制的特征拼接的准则:通常做通道间融合,例如:
  $A：[32,112,256,256];B:[32,32,256,256]$,$concat\_dim$为1,则输出的尺寸为 $[32,144,256,256]$
1.基本数据成员
```cpp
    int num_concats_; // 合并通道前的值，一般为mini——batch
    int concat_input_size_; //合并通道后的SIze一般为H*W
    int concat_axis_; // 合并的shape值，一般为1，即Channels合并
```

2.基本成员函数
  基本成员函数包含$LayerSetup$,$Reshape$,$Forward$,$Backward$,具体实现如下:
```cpp
// LayerSetup 主要包含参数的读取和判断是否合理
template <typename Dtype>
void ConcatLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>* >& bottom,
    const vector<Blob<Dtype>*>& top){
    const ConcatParameter& param = this->layer_param_.concat_param();
    // axis和concat_dim必须要有一个
    CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()));
}

// Reshape，根据prototxt的参数 ,决定top的shape
template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    const int num_axes = bottom[0]->num_axes();// 基本认为4 NCHW
    const ConcatParameter& param = this->layer_param_.concat_param();
    if (param.has_concat_dim()) {
        concat_axis_ = static_cast<int>(param.concat_dim());// 拼接的Channel
    }
    else{
        concat_axis_ = bottom[0]->CanonicalAxisIndex(param.axis());//默认C融合
    }

    vector<int>top_shape = bottom[0]->shape();//先初始一下bottom[0]--top_shape
    num_concats_ = bottom[0]->count(0,concat_axis_); //N;
    concat_input_size_ = bottom[0]->count(concat_axis_+1);//H*W
    int bottom_count_sum = bottom[0]->count();    // 输出count
    for (size_t i = 1; i < bottom.size(); i++) { // 决定输出的Size
        CHECK_EQ(bottom[i]->num_axes(),num_axes);//NCHW 四维
        for (size_t j = 0; j < num_axes; j++) {
            if (j == concat_axis_) {
                top_shape[j] += bottom[i]->shape(j);//拼接
            }
            //除了合并的通道shape要求可以不同，其余的都要相同
            CHECK_EQ(bottom[i].shape(j),top_shape[j]);
        }
        bottom_count_sum += bottom[i]->count();
    }
    top[0]->Reshape(top_shape);
    //其实没必要，这不出错则之前就会报错
    CHECK_EQ(bottom_count_sum,top[0]->count());
    if (bottom.size() == 1) {//类似于一个layer的拷贝
        top[0]->ShareData(*bottom[0]);
        top[0]->ShareDiff(*bottom[0]);
    }
}

// Forward_cpu 前向过程，比较简单，
// for循环完成整个的copy过程
template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    if (bottom.size() == 1) {
        return ;// Reshape 的时候完成了top的复制
    }
    Dtype* top_data = top[0]->mutable_cpu_data();
    int offset = 0; // 合并channel的偏移
    const int top_concat_axis = top[0]->shape(concat_axis_);//
    for (size_t i = 0; i < bottom.size(); i++) {
        const Dtype* bottom_data = bottom[i]->cpu_data();
        const int bottom_concat_axis = bottom[i]->shape(concat_axis_);//当前C
        for (size_t n = 0; n < num_concats_; n++) { //样本的循环
        //加入N*C*H*W,N循环，每次copy C*(H*W) 到top的 bottom_concat_axis*(H*W)
        // bottom是n*C_bottom*(H*W) top是( n*C_top + c)*(H*W)
            caffe_copy(bottom_concat_axis*concat_input_size_,//C_bottom*(H*W)
            bottom_data+n*bottom_concat_axis*concat_input_size_,
            top_data+(n*top_concat_axis+offset)*concat_input_size_);
        }
        offset += bottom_concat_axis;//处理完一个bottom,offset+C_bottom
    }
}

// BackFord_cpu过程，由于使用的是concat,输出只是输入的拼接，因此
// 只需要将top.diff 拆分为多块，每一块的bottom.diff对应top.diff
// offset 每次加上bottom的C
template <typename Dtype>
void ConcatLayer<Dtype>::BackFord_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>&propagate_down,const vector<Blob<Dtype>*>& bottom){
    if (bottom.size() == 1) {
        return ;// Reshape的室友ShareDiff已经copy
    }
    const Dtype* top_diff = cpu_diff(); // top层的loss
    int offset = 0;
    const int top_concat_axis = top[0]->shape(concat_axis_);//C_top
    for (size_t i = 0; i < bottom.size(); i++) {
        if (propagate_down[i]) {
            int bottom_concat_axis = bottom[i]->shape(concat_axis_);//C_bottom
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
            for (size_t n = 0; n < num_concats_; n++) { //样本遍历
                caffe_copy(bottom_concat_axis*concat_input_size_,//count
                top_diff+(n*top_concat_axis+offset)*concat_input_size_,
                bottom_diff+n*bottom_concat_axis*concat_input_size_);
            }
            offset += bottom_concat_axis;
        }
    }
}
```


##### $(5) \, SliceLayer$
  $SliceLayer$与$concat$是一个相反的过程，只不过$slice$是$bottom$分层，而$concat$是$bottom$的组合，通过$slice_point$来控制切片的格局,$axis$控制切片的通道.
1.基本数据成员
```cpp
    int num_slice_; //一般为N，即slice_axis的前面的乘积
    int slice_size_; //切成几片
    int slice_axis_; // NCHW哪一个开始切;
    vector<int>slice_point_;//prototxt的slice_point是134表示0-1 1-3 3-4
```
2.基本成员函数
  基本成员函数包含$LayerSetup$,$Reshape$,$Forward$,$Backward$,具体实现如下:
```cpp
    //LayerSetup 读取prototxt的参数
    template <typename Dtype>
    void SliceLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        const SliceParameter& param = this->layer_param_.slice_param();
        //类似concat axis或者 slice_dim prototxt中选一个
        CHECK(!(slice_param.has_axis() && slice_param.has_slice_dim()));
        slice_point_.clear();
        //其实就是把prototxt的slice_point参数push到slice_point_中
        //可以写成for i:slice_point_size(),slice_point_.push_back()
        std::copy(param.slice_point().begin(),
                param.slice_point().end(),std::back_inserter(slice_point_));
    }

    //Reshape ,top的size对应 slice_point_.size()+1 slice_point记录index
    template <typename Dtype>
    void SliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        const int num_axes = bottom[0]->num_axes();
        const SliceParameter& param = this->layer_param_.slice_param();

        //这里判断slice_dim的原因是，axis有default = 1
        if (param.has_slice_dim()) {
            slice_axis_ = static_cast<int>(param.slice_dim());
            //slice_axis_满足0----num_axes
        }
        else{
            slice_axis_ = bottom[0]->CanonicalAxisIndex(param.axis());//一般为1
        }

        //原始shape,后续只需要修改slice_axis_的shape即可
        vector<int>top_shape = bottom[0]->shape();
        const int bottom_slice_axis = bottom[0]->shape(slice_axis_);//切分的通道容量
        num_slice_ = bottom[0]->count(0,slice_axis_);//一般为N
        slice_size = bottom[0]->count(slice_axis_+1);//一般为H*W
        int count = 0;
        if (slice_point_.size() != 0) {
            CHECK_EQ(slice_point_.size(),top.size()-1);
            int prev = 0;
            vector<int> slices;//存放每个slice的Channels的大小
            for (size_t i = 0; i < slice_point_.size(); i++) {
                CHECK_GT(slice_point_[i],prev);//slice_point_的值要递增
                slices.push_back(slice_point_[i] - prev);
                prev = slice_point[i];
            }
            slices.push_back(bottom_slice_axis-prev);
            for (size_t i = 0; i < top.size(); i++) {
                top_shape[slice_axis_] = slices[i];
                top[i]->Reshape(top_shape);
                count += top[i]->count();
            }
        }
        // slice_point_ = 0,则根据top的size来进行均分
        else{
            CHECK_EQ(bottom_slice_axis % top.size(),0);//要整除
            top_shape[slice_axis_] = bottom_slice_axis / top.size();//每块的Channel
            for (size_t i = 0; i < top.size(); i++) {
                top[i]->Reshape(top_shape);
                count += top[i].count();
            }
        }
        CHECK_EQ(count,bottom[0]->count());//累加的top和bottom[0] count相同
        if (top.size() == 1) {
            top[0]->ShareData(*bottom[0]);// 类似于copy
            top[0]->ShareDiff(*bottom[0]);// 类似于copy
        }
    }

    // forward的过程，需要for top.size然后copy bottom的值
    // forward过程和concat的backward过程一致，只是diff-->data
    template <typename Dtype>
    void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        if (top.size() == 1) {
            return ;
        }
        int offset = 0; // 每次offset 加上 一个top的channels
        Const Dtype* bottom_data = bottom->cpu_data();
        const int bottom_slice_axis = bottom[0]->shape(slice_axis_);//总C_bottom
        for (size_t i = 0; i < top.size(); i++) {
            const int top_slice_axis = top[i]->shape(slice_axis_);//当前C_top
            Dtype* top_data = top[i]->mutable_cpu_data();
            for (size_t n = 0; n < num_slice_; n++) { //其实就是样本N
                caffe_copy(top_slice_axis*slice_size_, // 每次的copy_size C_TOP*H*W
                bottom+(n*bottom_slice_axis+offset)*slice_size_,//bottom地址
                top+n*top_slice_axis*slice_size_);//top的第n个样本的起始地址
            }
            offset += top_slice_axis;// 每次bottom的channels偏移一个top的C_top
        }
    }

    // backward 反向的top的梯度对应bottom的一部分，因此反向类似于
    // concat的前向，因此可以写出如下代码
    template <typename Dtype>
    void SliceLayer<Dtype>::BackFord_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
        if (top.size() == 1) {
            return ;
        }
        const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        int offset = 0;
        for (size_t i = 0; i < top.size(); i++) {
            if (propagate_down[i]) {
                const int top_slice_axis = top[i]->shape(slice_axis_);
                const Dtype* top_diff = top[i]->cpu_diff();
                for (size_t n = 0; n < num_slice_; n++) { // 样本数目
                    copy(top_slice_axis* slice_size_, // 每次copy的数据量
                        top_diff+n*top_slice_axis*slice_size_,//top的地址
                    bottom_diff+(n*bottom_slice_axis+offset)*slice_size_);
                }
                offset += top_slice_axis;
            }
        }
    }
```

##### $(6)\,FlatternLayer$
  $flatternLayer$实际完成输入维度的压缩，主要在$reshape$的操作上，
```cpp
    // Reshape 64*10*30*30 axis = 1，end_axis =2则 10*300*30
    template <typename Dtype>
    void FlatternLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
      const FlatternParameter& param = this>layer_param_.flattern_param();
      const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
      const int end_axis = bottom[0]->CanonicalAxisIndex(param.end_axis());
      vector<int>top_shape;
      for (size_t i = 0; i < start_axis; i++) {
        top_shape.push_back(bottom[0]->shape(i));//前面的shape保持不变
      }
      const int flattern_dim = bottom[0]->count(start_axis,end_axis+1);
      top_shape.push_back(flattern_dim);
      for (size_t i = end_axis+1; i < bottom[0].num_axes(); i++) {
        top_shape.push_back(bottom[0]->shape(i));
      }
      top[0]->Reshape(top_shape);
    }
    //forward 和 backward同正常的feedforwad相同 HCHW展开式相同的
    top[0]->ShareData(*bottom[0]);
    bottom[0]->ShareDiff(*top[0]);
```

##### $(7)\, DropoutLayer$
  $DropoutLayer$在深度学习的网络结构中对网络过拟合起到很大的作用，通过设置$drop_ratio$来控制网络的结点开闭，从而产生网络的异构多样性，降低网络的过拟合。
1.基本数据成员
  在训练阶段，结点是随机开闭的，但是在预测阶段，结点均开，但是输出会乘以概率p
```cpp
    Blob<unsigned int> rand_vec_;//存放bottom对应位置随机出来的值
    Dtype threshold_;// drop的阈值
    Dtype scale_ ;// scale因子,由于结点闭合的原因，开放的结点需要乘以的因子
    // 这里判断是否参数训练的时候乘以了scale,结点少了每个结点权重提高
    bool scale_train_;
```
2.基本成员函数
```cpp

    //LayerSetup,类似于activation，读取参数
    template <typename Dtype>
    void DropoutLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>* >& top){
      NeuronLayer<Dtype>::LayerSetUp(bottom, top);
      const DropoutParamter& param = this->layer_param_.dropout_param();
      threshold_ = param.dropout_ratio();
      scale_ = 1./(1-threshold_);// 测试的时候开放的scale
      scale_train_ = param.scale_train();
      unit_thres_ = static_cast<unsigned int>(UINT_MAX*threshold_);
    }

    // Reshape
    // 类似于激励函数，只是有的toplayer会闭合置零
    template <typename Dtype>
    void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>* >& top){
      NeuronLayer<Dtype>::Reshape(botom,top);
      rand_vec_.Reshape(bottom[0]->shape());
    }


    template <typename Dtype>
    void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>* >& top){
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const int count = bottom[0]->count();
      Dtype top_data = top[0]->mutable_cpu_data();
      if (this->phase_ == "TRAIN") { // 如果是训练阶段
         cafe_rng_bernoulli(count,1.-threshold_,rand_vec_.mutable_cpu_data());
         if (scale_train_) { //训练时是否每个结点都提高权重
           for (size_t i = 0; i < count; i++) {
             //rand_vec_为1表示保留，为0表示闭合drop
             top_data[i] = bottom_data[i]*rand_vec_.cpu_data()[i]*scale_;
           }
         }
         else{
           for (size_t i = 0; i < count; i++) {
             top_data[i] = bottom_data[i]*rand_vec_.cpu_data()[i];
           }
         }
      }
      //测试阶段全开，如果训练提高权重则不处理，反之则测试的时候除以权重
      else{
          caffe_copy(count,bottom_data,top_data);
          if (!scale_train_) {
             caffe_scal<Dtype>(count,1./scale,top_data);
          }
      }
    }

    // backward_cpu过程,开的才会有偏导，根据rand_vec_的值来确定
    template <typename Dtype>
    void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,const vector<Blob<Dtype>* >& bottom){
      const int count = bottom[0]->count();
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      if (propagate_down[0]) {
        const unsigned int* mask = rand_vec_->cpu_data();
        if (this->phase_ == "TRAIN") {
            if (scale_train_) {
              for (size_t i = 0; i < count; i++) {
                  bottom_diff[i] = top_diff[i]*mask[i]*scale_;
              }
            }
            else{//训练的时候没有乘以扩增因子
              for (size_t i = 0; i < count; i++) {
                  bottom_diff[i] = top_diff[i]*mask[i];
              }
           }
        }
      // 测试的时候
      else{
        caffe_copy(count,top_diff,bottom_diff);
        if (!scale_train_) {
          caffe_scal<Dtype>(count,1./scale_,bottom_diff);
        }
      }
    }
  }
```





>本文作者： 张峰
>本文链接：[https://zhanglaplace.github.io/2017/10/20]( https://zhanglaplace.github.io/2017/10/20/Caffe%20Loss%E5%88%86%E6%9E%90/)
>版权声明：本博客所有文章，均采用CC BY-NC-SA 3.0 许可协议。转载请注明出处！

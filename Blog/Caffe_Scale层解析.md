---
title: Caffe Scale层解析
date: 2017-11-09 09:26:33
tags: [Caffe,DeepLearning]
categories: Caffe
---

# Caffe Scale层解析
  前段时间做了caffe的batchnormalization层的解析，由于整体的BN层实现在Caffe是分段实现的，因此今天抽时间总结下Scale层次，也会后续两个层做合并做下铺垫。
<!--more-->
## 基本公式梳理
  Scale层主要完成 $top = alpha*bottom+ beta$的过程，则层中主要有两个参数$alpha$与$beta$,
  求导会比较简单。
  $$ \frac{\partial y}{\partial x} = alpha ;\quad \frac{\partial y}{\partial alpha} = x;\quad \frac{\partial y}{\partial beta} = 1$$
  需要注意的是$alpha$与$beta$均为向量，针对输入的$channels$进行的处理，因此不能简单的认定为一个$float$的实数。

## 具体实现
  该部分将结合源码实现解析$scale$层:
  在Caffe proto中ScaleParameter中对Scale有如下几个参数：
```cpp
  axis [default = 1] ; 默认的处理维度
  num_axes [default = 1] ; //在BN中可以忽略，主要决定第二个bottom
  FillerParameter filler ; //两个FillerParameter即决定初始alpha和beta的填充方式。
  //决定是否学习bias，如果不学习，则可以简化为alpha*x = y
  optional bool bias_term = 4 [default = false];
  FillerParameter bias_filler;
```
### 基本成员变量
```cpp
  // caffe的scale层实现+beta调用了bias层。。。。。。。。。。
  shared_ptr<Layer<Dtype>> bias_layer_; /
  vector<Blob<Dtype>*>bias_bottom_vec_;
  vector<bool> bias_propagate_down_;
  int bias_param_id_;

  Blob<Dtype> sum_multiplier_;
  Blob<Dtype>sum_result_;
  Blob<Dtype> temp_;
  int axis_;
  int outer_dim_,inner_dim_,scale_dim_;
```
 基本成员变量主要包含了Bias层的参数以及Scale层完成对应通道的标注工作。

 ### 基本成员函数
 主要包含了LayerSetup,Reshape ,Forward和Backward ，内部调用的时候bias_term为true的时候会调用biasLayer的相关函数.

 #### LayerSetup,层次的建立
 ```cpp
 template <typename Dtype>
 void ScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
     const ScaleParameter param = this->layer_param_->scale_param();
     if (bottom.size() == 1 && this->blobs_.size() > 0) {
       //区分测试与训练，测试时 blobs-已经有值
     }
     else if(bottom.size() == 1){
        // 考虑BN的scale 不需要考虑axes
        axis_ = bottom[0]->CanonicalAxisIndex(param.axis());// 1 通道
        const int num_axes = param.num_axes(); // 1
        this->blobs_.resize(1);// alpha;

        //这么大一串，实际就是blobs_[0].reset(new Blob<Dtype>(vector<int>(C)));
        const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis_;
        const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
        vector<int>scale_shape(shape_start, shape_end);
        this->blobs_[0].reset(new Blob<Dtype>(scale_shape));

        FillerParameter filler_param(param.filler());
        if (!param.has_filler()) { //没写明填充模式
          filler_param.set_type("constant");
          filler_param.set_value(1);
        }
        shared_ptr<Filler<Dtype>>filler(GetFiller<Dtype>(filler_param));
        filler->Fill(this->blobs_[0].get());
     }

     // 处理需不需要bias
     if (param.bias_term()) {
       LayerParameter layer_param(this->layer_param_);
       layer_param.set_type("Bias");
       BiasParameter* bias_param = layer_param_.mutable_bias_param();
       bias_param->set_axis(param.aixs());
       if (bottom.size() > 1) {
          bias_param->set_num_axes(bottom[1]->num_axes());
       }
       else{
         bias_param->set_num_axes(param.num_axes());//bn层走下面
       }
       bias_param->mutable_filler()->CopyFrom(param.bias_filler());
       bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
       bias_bottom_vec_.resize(1);
       bias_bottom_vec_[0] = bottom[0];
       bias_layer_->Setup(bias_bottom_vec_,top);
       bias_param_id = this->blobs_.size(); //1 alpha 此处增加个beta
       this->blobs_.resize(bias_param_id_+1); // 2
       this->blobs_[bias_param_id] = bias_layer_->blobs()[0];
       bias_propagate_down_.resize(1,false);
     }
     this->param_propagate_down_.resize(this->blobs_.size(),true);
 }
 ```
 Scale层的一部分在完整BN中是不需要考虑的，完整BN中bottomSize为1，num_axes默认为1，blobs_[0]为长度为C的向量，bias需要调用caffe的bias层，所以会看着比较麻烦。

#### Reshape 调整输入输出与中间变量
  Reshape层完成许多中间变量的size初始化
```cpp
//Reshape操作
template <typename Dtype>
void ScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const ScaleParameter param = this->layer_param_->scale_param();
    Blob<Dtype>* scale = (bottom.size() > 1)?bottom[1]:this->blobs_[0].get();
    axis_=(scale->num_axes()==0)?0:botom[0]->CanonicalAxisIndex(param.axis());
    //这里做了下比较 bottom的NCHW axis_ = 1 则 C == C
    CHECK_EQ(bottom[0]->shape(axis_) = scale->shape(0));

    outer_dim_ = bottom[0]->count(0,axis_);// n
    scale_dim = scale->count(); //c
    inner_dim_ = bottom[0]->count(axis+1);// hw

    if (bottom[0] == top[0]) {
      // Layer得top和bottom同名 in-place computation
      const bool scale_param = (bottom.size() == 1); //true
      if (!scale_param || (scale_param && this->param_propagate_down_[0]) {
        // 后面一个条件成立，需要backward
        //防止修改top时，bottom改变，做临时,因为求导要用到原始的bottom-data
        temp_.ReshapeLike(*bottom[0]);
      }
    }
    else{
      top[0]->ReshapeLike(*bottom[0]);//
    }
    //类似于bn的num-by—tran 保存中间的NC结果 NC*1*1*1
    sum_result_.Reshape(vector<int>(1,outer_dim_*scale_dim_));
    const int sum_mult_size = std::max(outer_dim_,inner_dim_);
    // 为什么不类似于BN做两个temp vector呢
    sum_multiplier_.Reshape(vector<int>(1,sum_mult_size));
    if (sum_multiplier_.cpu_data()[sum_mult_size-1] != Dtype(1)) {
      caffe_set(sum_mult_size,Dtype(1),sum_multiplier_.mutable_cpu_data());
    }
    if (bias_layer_) {
      bias_bottom_vec_[0] = top[0];
      bias_layer_->Reshape(bias_bottom_vec_,top);
    }
}
```
  Reshape操作同BN的基本相似，只不过此处只是新建了两个中间变量，sum_multiplier_和sum_result_.

#### Forward 前向计算
  前向计算，在BN中国紧跟着BN的归一化输出，完成乘以alpha与+bias的操作，由于alpha与bias均为C的向量，因此需要先进行广播。
```cpp
template <typename Dtype>
void ScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (bottom[0] == top[0]) {
     // 先进行一次临时拷贝复制
     caffe_copy(bottom[0]->count(),bottom_data,temp_.mutable_cpu_data());
  }
  const Dtype* scale_data = (bottom.size() > 1)?bottom[1]:
                      this->blios_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // 这里的遍历实际上和广播的类似，一种是每次操作inner_dim个元素，一种是讲alpha
  // 广播到整个feature_map，然后再调用一次cpu_scale
  for (size_t n = 0; n < outer_dim_; n++) { // n
    for (size_t d = 0; d < scale_dim_; d++) { //c
       const Dtype factory = scale_data[d];// 取某一个通道的值
       caffe_cpu_scale(inner_dim,factory,bottom_data,top_data);
       top_data += inner_dim_;
       bottom_data += inner_dim;
    }
  }
  if (bias_layer_) {
     bias_layer_->Forward(bias_bottom_vec_,top);
  }
}
```

#### Backward 反向计算
  主要求解三个梯度，对alpha 、beta和输入的bottom(此处的temp)
  ```cpp
  template <typename Dtype>
  void ScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (bias_layer_ &&  // 默认false
      this->param_propagate_down_[this->param_propagate_down_.size()-1]) {
      bias_layer_->Backward(top,bias_propagate_down_,bias_bottom_vec_);
    }
    const scale_param = (bottom.size() == 1);
    Blob<Dtype>* scale = scale_param? this->blobs_[0].get(),bottom[1];
    if ((!scale_param && propagate_down[1])|| //bottomsize大于1的时候判断
      (scale_param&&this->param_propagate_down_[0])) {// 1个输入是判断alpha
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const in_place = (bottom[0] == top[0]);
      // 需要做备份 如果输入输出同名，需要注意用原来临时的temp
      const Dtype* bottom_data =in_place?temp_.cpu_data():bottom[0]->cpu_data();
      // BN中输入是NCHW,而alpha和beta仅仅针对C
      const bool is_eltwise = (bottom[0]->count() == scale->count());//不相等的
      Dtype* product= is_eltwise?scale_.mutable_cpu_diff():
      (in_place?temp_.mutable_cpu_data():bottom[0]->mutable_cpu_diff());
      caffe_mul(top[0]->count(),top_diff,bottom_data,product);
      if (!is_eltwise) { // blobs_与输入对不上
        Dtype* sum_result_ = NULL;
        if (inner_dim_ == 1) {
          //H*W == 1;
          sum_result_ = product;
        }
        else if(sum_result_.count() == 1){ // 1*1*1*1
          const Dtype* sum_mult_  = sum_multiplier_.cpu_data();
          Dtype* scale_diff = scale->mutable_cpu_diff();
          if (scale_param) { //true
            Dtype result = caffe_cpu_dot(inner_dim,product,sum_mult);
            *scale_diff += result; //H*W的相乘
          }
          else{
            *scale_diff = caffe_cpu_dot(inner_dim_,product,sum_mult);
          }
        }
        else{
          const Dtype* sum_mult = sum_multiplier_.mutable_cpu_data();
          sum_result = (outer_dim_ == 1)? // nc如果n==1就直接幅值C
          scale_.mutable_cpu_diff():sum_result_.mutable_cpu_data();

          //NC HW  * HW*1 = NC*1 HW全1
          caffe_cpu_gemv<Dtype>(CblasNoTrans,sum_result.count(),inner_dim,
          Dtype(1),product,sum_mult,Dtype(0),Dtype(0),sum_result);
        }

        if (out_dim_ != 1) {
           const Dtype* sum_mult  = sum_multiplier_.cpu_data();
           Dtype* scale_diff = scale->mutable_cpu_diff();
           if (scale_dim_ ==1) {
             if (scale_param) { // C==1直接计算 NC*NC
                Dtype result = caffe_cpu_dot(outer_dim_,sum_mult_,sum_result);
                *scale_diff += result;
             }
             else{
               *scale_diff =  caffe_cpu_dot(outer_dim_,sum_mult_,sum_result);
             }
           }
           else{  //如果C != 1 需要gemv,(num * channels)^t * 1 *num*1
             caffe_cpu_gemv<Dtype>(CblasTrans,outer_dim_,scale_dim,
             Dtype(1),sum_result,sum_mult,Dtype(scale_param),scale_diff);
           }
        }
    }
  }
  if (propagate_down[0]) { //x求导
     const Dtype* top_diff = top[0]->cpu_diff();
     const Dtype* scale_data = scale->cpu_data();
     Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
     for (size_t n = 0; n < outer_dim_; n++) {
       for (size_t d = 0; d < inner_dim_; d++) {
          const Dtype factory = scale_data[d];
          caffe_cpu_scale(inner_dim_,factory,top_diff,bottom_diff);
          bottom_diff += inner_dim_;
          top_diff += inner_dim_;
       }
     }
  }
}
  ```
Caffe中的Scale层由于不仅仅作为BN的后续层，因此看着会比较绕，实际去上去掉很多if else 后会清晰很多








 >本文作者： 张峰
 >本文链接：[http://www.enjoyai.site/2017/11/09](http://www.enjoyai.site/2017/11/09/Caffe_Scale%E5%B1%82%E8%A7%A3%E6%9E%90/ )
 >版权声明：本博客所有文章，均采用CC BY-NC-SA 3.0 许可协议。转载请注明出处！

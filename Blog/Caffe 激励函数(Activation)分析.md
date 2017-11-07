---
title: Caffe NeuronLayer分析
date: 2017-10-20 11:16:01
tags: [Caffe,DeepLearning]
categories: Caffe
---

### Caffe_NeuronLayer
  一般来说，激励层的输入输出尺寸一致，为非线性函数，完成非线性映射，从而能够拟合更为复杂的函数表达式激励层都派生于NeuronLayer: class XXXlayer : public NeuronLayer<Dtype>
#### 1.基本函数
  激励层的基本函数较为简单，主要包含构造函数和前向、后向函数
```cpp
  explicit XXXLayer(const LayerParameter& param)
          :NeuronLayer<Dtype>(param){}
  virtual inline const char* type() const { return "layerNane"; }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
```
<!--more-->
#### 2.常用$Neuron$层

##### (1) Relu/PRelu Rectufied Linear Units
   ReLU的函数表达式为 $f(x) = x\*(x>0) + negative\_slope\*x\*(x <= 0)$ 具体实现如下
```cpp
  //forward_cpu
  template <typename Dtype>
  void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>& top){ // 根据bottom求解top
      const Dtype* bottom_data = bottom[0]->cpu_data();//const 不可修饰
      Dtype* top_data = top[0]->mutable_cpu_data();//可修饰
      const int count = bottom[0]->count();//因为count_一致，也可用top
      Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
      for (size_t i = 0; i < count; i++) {
         top_data[i] = bottom_data[i]*(bottom_data[i] > 0)
                    + negative_slope*bottom_data[i]*(bottom_data[i] <= 0);
      }
  }


  //Backward_cpu
  // 导数形式 f'(x) = 1 x>0 ; negative_slope*x x<0
  template <typename Dtype>
  void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
      const Dtype* top_diff = top[0].cpu_diff();//top diff
      const Dtype* bottom_data = bottom[0].cpu_data();//用以判断x是否大于0
      Dtype* bottom_diff = bottom[0].cpu_diff();//bottom diff
      const int count = bottom[0].count();
      for (size_t i = 0; i < count; i++) {
         bottom_diff[i] = top_diff[i]*(bottom_data[i] > 0)
                    +negative_slope*(bottom_data[i] <= 0);
      }
  }

// Relu 函数形式简单，导函数简单，能有效的解决梯度弥散问题，但是当x小于0时，易碎
// 但是网络多为多神经元，所以实际应用中不会影响到网络的正常训练。
```

##### (2) Sigmoid (S曲线)
   Sigmoid函数表达式为$f(x) = 1./(1+exp(-x))$;值域0-1，常作为BP神经网络的激活函数
由于输出为0-1，也作为logistic回归分析的概率输出函数。具体实现如下;
```cpp

    //定义一个sigmoid函数方便计算
    template <typename Dtype>
    inline Dtype sigmoid(Dtype x){
       return 1./(1.+exp(-x));
    }
    //前向 直接带入sigmoid函数即可
    template <typename Dtype>
    void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();//需要计算
        const int count = bottom[0]->count();//N*C*H*W;
        for (size_t i = 0; i < count; i++) {
           top_data[i] = sigmoid(bottom_data[i]);
        }
    }

    //Backward_cpu 由于f'(x) = f(x)*(1-f(x))，所以需要top_data
    // bottom_diff = top_diff*f'(bottom_data) = top_diff*top_data*(1-top_data)
    template <typename Dtype>
    void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* top_data = top[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff(); //需要计算
        const int count = bottom[0]->count();
        for (size_t i = 0; i < count; i++) {
            //top_data[i] == sigmoid(bottom_data[i]);
            bottom_diff[i] = top_diff[i]*top_data[i]*(1.-top_data[i]);
        }
    }

// Sigmoid函数可以作为二分类的概率输出，也可以作为激活函数完成非线性映射，但是网络
// 增加时，容易出现梯度弥散问题，目前在CNN中基本不使用

```

##### (3)TanH,双正切函数
  TanH函数的表达式为 $f(x) =\frac{(1.-exp(-2x))}{(1.+exp(-2x))}$;值域0-1,与sigmoid函数有相同的问题,
但是TanH在RNN中使用较为广泛,[理由参考](https://www.zhihu.com/question/61265076/answer/186644426)，具体实现如下所示。

```cpp
    //定义一个tanH的函数表达式,实际已经封装
    inline Dtype TanH(Dtype x){
       return (1.-exp(-2*x))/(1.+exp(-2*x));
    }

    //Forward_cpu
    template <typename Dtype>
    void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        for (size_t i = 0; i < count; i++) {
            top[i] = TanH(bottom_data[i]);
        }
    }

    //Backward_cpu f'(x) = 1-f(x)*f(x);
    // bottom_diff = top_diff(1-top_data*top_data);
    template <typename Dtype>
    void TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* top_data = top[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff(); //需要计算
        const int count = bottom[0]->count();
        for (size_t i = 0; i < count; i++) {
            //top_data[i] == TanH(bottom_data[i]);
            bottom_diff[i] = top_diff[i]*(1.-top_data[i]*top_data[i]);
        }
    }
```

    其他的激励函数就不在枚举，可以查看具体的caffe源码，实现大致相同

### 3.说明
#### (1) 梯度弥散和梯度爆炸
  网络方向传播时，loss经过激励函数会有$loss*\partial{f(x)}$,而如sigmoid的函数，
max($\partial{f(x)}$)只有1/4因此深层网络传播时loss越来越小，则出现前层网络未完整学习而后层网络学习饱和的现象

#### (2) Caffe激励层的构建
  如上述的代码所示，激励层主要完成forward和Bacward的函数实现即可，由构建的函数表达式推导出它的导函数形式，弄懂bottom_data,top_data,bottom_diff,top_diff即可
  >本文作者： 张峰
  >本文链接：[https://zhanglaplace.github.io/2017/10/20]( https://zhanglaplace.github.io/2017/10/20/Caffe%20%E6%BF%80%E5%8A%B1%E5%87%BD%E6%95%B0(Activation)%E5%88%86%E6%9E%90/)
  >版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！

---
title: Caffe Loss分析
date: 2017-10-20 19:16:01
tags: [Caffe,DeepLearning]
categories: Caffe
---

### Caffe_Loss
  损失函数为深度学习中重要的一个组成部分，各种优化算法均是基于Loss来的，损失函数的设计好坏很大程度下能够影响最终网络学习的好坏。派生于 $LossLayer$,根据不同的Loss层有不同的参数;
#### 1.基本函数
  主要包含构造函数，前向、后向以及Reshape，部分有SetUp的函数，每层都有Loss参数
```cpp
    explicit XXXLossLayer(const LayerParameter& param):
    LossLayer<Dtype>(param),diff_() {}
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
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

#### 2.常用损失函数
  通常在训练过程中，采用mini_batch的方式
##### (1) EuclideanLoss (欧式损失函数，L2损失)
  $EuclideanLoss$的公式表达为 $loss = \frac{1}{2n}\sum_{i=1}^n{(y_{i}-\hat{y}_{i})^2}$
```cpp
  //reshape函数，完成层次的reshape,diff_与输入的N*C维度相同
  template <typename Dtype>
  void EuclideanLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
      LossLayer<Dtype>::Reshape(bottom,top);//先调用基类的Reshape函数
      CHECK_EQ(bottom[0]->count(1),bottom[1]->count(1));//label类别
      diff_.Reshape(*bottom[0]);//一般是N*C*1*1
  }

  // Forward_cpu 前向 主要计算loss
  template <typename Dtype>
  void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
     const int count = bottom[0]->count();
     caffe_sub(count,
               bottom[0]->cpu_data(),//网络的输出 N*C
               bottom[1]->cpu_data(),//对应label N*C
               diff_.mutable_cpu_data()//对应的loss差分
           );//完成 y_{predicy}-y_{label} //bottom[0]-bottom[1]
     Dtype dot = caffe_cpu_dot(count,diff_.cpu_data(),diff_.cpu_data());
     //bottom[0]->num()== bottom[0].shape(0);
     Dtype loss = dot/bottom[0]->num()/Dtype(2);//loss/(2*n)
     top[0]->mutable_cpu_data()[0] = loss;
  }

 //Backward_cpu f'(x) = 1/n*(y_{predict}-y_{label})
 template <typename Dtype>
 void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>&propagate_down,const vector<Blob<Dtype>*>& bottom){
    for (size_t i = 0; i < 2; i++) {
        if (propagate_down[i]) {//需要backward
            //对应predict-label 如果label为bottom[0]就需要乘以-1
            const Dtype sign = (i==0) ? 1 : -1;
            //top[0]->cpu_diff()返回float* length = 1;下式为loss/n;
            const Dtype alpha = sign*top[0]->cpu_diff()[0]/bottom[0]->num();
            //y = ax+by ;
            caffe_cpu_axpby(bottom[0]->count(),//count
                            alpha,// loss/n
                            diff_.cpu_data(),//y_{predict}-y_{label}
                            Dtype(0),
                            bottom[i]->mutable_cpu_diff()
                        );//1/n*loss*(y_{predict}-y_{label})
        }
    }
    //欧式损失函数形式简单，常用于做回归分析，做分类需要统一量纲。
 }
```

##### (2)SoftmaxWithLoss Softmax损失函数
$\qquad softmax函数将输出的各个类别的概率值进行归一化，生成各个类别的prob$
$\qquad 常用的分类损失函数，Softmax输出与Multinomial Logistic Loss的结合。公式如下:$
$$ y_i = softmax(x_i) = \frac{exp(x_i)}{\sum_{j=1}^{n}{exp(x_j)}}$$
$$loss = -log(y_k) ,k为实际的样本label$$
$\qquad 损失函数的推导:\frac{\partial Loss}{\partial x_i}=\sum_{j=1}^{n}{\frac{\partial loss}{\partial y_j}\*\frac{\partial y_j}{\partial x_i}}=-\frac{1}{y_k}\*\frac{\partial y_k}{\partial x_i} \quad k为实际的label,其他的\frac{\partial loss}{\partial y_j} =0 \\$
$$
\qquad \frac{\partial y_k}{\partial x_i} = \frac{\partial softmax(x_k)}{\partial x_i}=
\begin{cases}
\  y_k\*(1-y_k) \qquad k == i \\\
\\
\ -y_k*y_i \qquad \qquad k \,\,!=\,i
\end{cases}
$$
$$
整理后可以发现\frac{\partial loss}{\partial x_i}=
\begin{cases}
\  y_k-1 \qquad k \,== \,i ，即i为实际label\\\
\\
\  y_i \qquad \qquad k \,\,!=\,i,即i不是实际label
\end{cases}
$$
    具体代码的实现如下所示:
1.SoftmaxWithLossLayer的输入:bottom
```cpp
    // bottom[0]为前层的特征输出，一般维度为N*C*1*1
    // bottom[1]为来自data层的样本标签，一般维度为N*1*1*1;
    // 申明
    const vector<Blob<Dtype>*>& bottom;
    //backward部分代码
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();//label
```
2.SoftmaxWithLossLayer层的输出:top
```cpp
    // SoftmaxWithLossLayer的输出其实就是1*1*1*1的最终loss
    // 如果有多个的话实际就是也会保存softmax的输出，但是需要注意的是内部包含了
    //Softmax的FORWAR过程，产生的概率值保存在prob_内
    const vector<Blob<Dtype>*>& top;
    //forward部分代码 ,
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    if (top.size() == 2) {
        top[1]->ShareData(prob_);//top[1]保存softmax的前向概率
    }
```
3.SoftmaxWithLossLayer的关键变量: $softmax\_top\_vec\_,prob\_$ 记录中间值
```cpp
    shared_ptr<Layer<Dtype> > softmax_layer_;
    /// prob stores the output probability predictions from the SoftmaxLayer.
    Blob<Dtype> prob_;
    /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
    vector<Blob<Dtype>*> softmax_bottom_vec_;
    /// top vector holder used in call to the underlying SoftmaxLayer::Forward
    vector<Blob<Dtype>*> softmax_top_vec_;
    /// Whether to ignore instances with a certain label.
    bool has_ignore_label_;
    /// The label indicating that an instance should be ignored.
    int ignore_label_;
    /// How to normalize the output loss.
    LossParameter_NormalizationMode normalization_;

    int softmax_axis_, outer_num_, inner_num_;//softmax的输出与Loss的维度
    template <typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        LossLayer<Dtype>::Reshape(bottom,top);//先调用基类的reshape
        softmax_layer_->Reshape(softmax_bottom_vec,softmax_top_vec_);
        int axis = this->layer_param_.softmax_param().axis();//softmaxproto参数(1)
        softmax_axis_ = bottom[0]->CanonicalAxisIndex(axis);//正不变负倒数
        outer_num_ = bottom[0]->count(0,softmax_axis_);// N mini_batch_size
        inner_num_ = bottom[0]->count(softmax_axis_+1);// H*W 一般为1*1
        //保证outer_num_*inner_num_ = bottom[1]->count();//bottom[1]为label N
        if (top.size() >= 2) {//多个top实际上是并列的，prob_值完全一致
            top[1]->Reshapelike(*bottom[0]);
        }
    }

    //forward是一个计算loss的过程，loss为-log(p_label)
    //由于softmaxWithLoss包含了Softmax所以需要经过Softmax的前向，并得到每个类别概率值
    template <typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        //调用Softmax的前向
        softmax_layer_->Forward(softmax_bottom_vec_,softmax_top_vec_);
        //这里等同于softmax_top_vec_[0]->cpu_data();
        const Dtype* prob_data = prob_.cpu_data();
        const Dtype* label = bottom[1]->cpu_data();//label 一般来自Data层
        // 一般是N*C(n个样本，每个C个预测概率)/ N == 类别数目
        int dim = prob_.count()/out_num_;
        int count = 0;//统计实际参与loss的样本个数
        Dtype loss = 0;
        for (size_t i = 0; i < outer_num_; i++) {//每个样本遍历
            for (size_t j = 0; j < inner_num_; j++) { //可以认为j == 0 绝大多数成立
                const int label_value = static_cast<int>(label[i*inner_num_+j]);
                if(has_ignore_label_ && label_value == ignore_label_){
                    // softmaxLayer的参数，可以选择不参与loss的类别
                    continue;
                }
                else{//实际需要判断label_value > 0 ,< prob_.shape(1)
                    // -= 因为loss = -log(p_label),prob_data 是n*c的
                    loss -= log(std::max(prob_data[i*dim+label_value*inner_num_+j)],
                                    Dtype(FLT_MIN)));//防止溢出或prob出现NAN
                    ++count;
                }
            }
        }
        //全部样本遍历完成后，可以进行归一，其实也挺简单，
        // top[0]->mutable_cpu_data[0] = loss/归一化
    }

    // Backward_cpu,这里的Backward实际需要更新的是softmax的输入接口的数据，
    // 中间有个y的转化，具体公式上面已经写出
    // bottom_diff = top_diff * softmaxWithloss' = top_diff * {p -1 或者 p}
    template <typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
        //fc输出与label的位置固定了，因此不需要如同欧式loss去判断label和fc的输入位置
        if (propagate_down[1]) {
            //label不需要backpropagate
        }
        if (propagate_down[0]) {//输入，需要更新
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();//需要修改的
            const Dtype* prob_data = prob_.cpu_data();//N*C
            //这里把diff先确定为softmax输出的y值，即bottom_diff[t] = y_t ;
            caffe_copy(prob_.count(),prob_data,bottom_diff);
            const Dtype* label = bottom[1]->cpu_data();
            // 也可以替换为bottom[1]->count(),实际就是类别C
            int dim = prob_.count()/ outer_num_;//NC/C == N
            int count = 0;
            for (size_t i = 0; i < outer_num_; i++) { //n个样本
                for (size_t j = 0; j < inner_num_; j++) { // 实际j == 0
                    const int label_value = static_cast<int>(label[i*inner_num_+j]);
                    if (has_ignore_label_ && label_value == ignore_label_) {
                        //正好是忽略loss的类别
                        bottom_diff[i*dim+label_vale*inner_num_+j] = 0;
                    }
                    else{
                        //这里需要考虑为什么，实际上之前所有的diff初始为y_t，
                        //根据softmax的偏导知道真实label是y_t -1;
                        bottom_diff[i*dim+label_vale*inner_num_+j] -= 1;
                        ++count;
                    }
                }
            }
            //这里只完成了loss的一部分，还差top_diff即Loss
            //如果归一化，就进行归一，同cpu_forward
            //cpu_diff可以认为是Loss
            // Dtype loss_weight = top[0]->cpu_diff()[0]/归一化
            caffe_scal(prob_count(),loss_weight,bottom_diff);
        }
    }

```

##### (3) SmoothL1Loss (RCNN提出的Loss)
  $SmoothL1Loss$为欧式均方误差的修改版，为分段函数，对离散点不敏感,具体的公式如下:
$$
SmoothL1Loss(x) =
\begin{cases}
\  0.5\*(sigma\*x)^2 \qquad 其他
\\
\  \left|x\right|-0.5/sigma^2 \qquad \left|x\right| < 1./sigma^2
\end{cases}
$$
整体的公式为:$x_{new} = x_{input}\*w_{in},output = w_{out}\*SmoothL1loss(x_{new});$
1.基本的数据类型和意义:
```cpp
    Blob<Dtype> diff_;// y_
    Blob<Dtype> error_;//loss
    Blob<Dtype> ones_;
    bool has_weights_; // weight权值
    Dtype sigma2_ ;// sigma 默认为1，此处sigma2_ = sigma*simga;
```
2.基本的功能函数
    基本包含了LayerSetup Reshape Forward 和 Backward四个函数,具体实现如下
```cpp
    //构建layer层次,SmoothL1LossLayer的参数有sigma，默认为1
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>&bottom,
    const vector<Blob<Dtype>*>& top){
        SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
        sigma2_ = loss_param.sigma()*loss_param.sigma();
        has_weights_ = (bottom.size() >= 3);//bottom[3]---为weights
        if (has_weights_) {
            //bottom[3] == out_weight;//w_out
            //bottom[2] == in_weight;// w_in
        }
    }

    // Reshape 根据输入输出调节结构，计算过程进行了拆分
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>&
        bottom,const vector<Blob<Dtype>*>& top){
        LossLayer<Dtype>::Reshape(bottom,top);//基函数
        //这里判断参数维度,
        if (has_weights_) {
            CHECK_EQ(bottom[0]->count(1) == bottom[2]->count(1) ==
            bottom[3].count(1))  ;//w_in和w_out的权值
        }
        diff_.Reshape(bottom[0].shape());// diff_ = w_in*(bottom[0]-bottom[1]);
        error_.Reshape(bottom[0].shape());// error_ = w_out*smoothL1(w_in*diff_);
        ones_.Reshape(bottom[0].shape());// one_ = error_*w_out;
        for (size_t i = 0; i < ones_->count(); i++) {
            one_s.mutable_cpu_data()[i] = Dtype(1);
        }
    }

    // Forward过程，一步一步操作
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        int count = bottom[0]->count();
        //bottom[0]和bottom[1]不确定标签和特征的顺序
        caffe_gpu_sub( // 计算diff_ = bottom[0]-bottom[1];
            count,
            bottom[0]->gpu_data(),
            bottom[1]->gpu_data(),
            diff_.mutable_cpu_data()
        );
        if (has_weights_) { x_new = x_input*in_weight,xinput==diff_
            caffp_gpu_mul(
                count,
                bottom[2]->gpu_data(),
                diff_.gpu_data(),
                diff_.mutable_gpu_data();
            );
        }
        //此处为SmoothL1的函数前向过程GPU实现
        SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
            count, diff_.gpu_data(), errors_.mutable_gpu_data(), sigma2_);
        CUDA_POST_KERNEL_CHECK;

        if (has_weights_) { //x_out= SmoothL1(w_in*x_input) * w_out
            caffe_gpu_mul(
                count,
                bottom[3]->gpu_data(),
                error_.gpu_data(),
                error_.mutable_gpu_data();
            ); // error _ = w_out* error_
        }
        Dtype loss;
        caffe_gpu_dot(count,ones_.gpu_data().error_gpu_data(),&loss);//类似于asum
        top[0]->mutable_gpu_data()[0] = loss/bottom[0]->num();// mini_batch
    }

    // GPU的实现SmoothL1loss,根据公式实现即可
    template <typename Dtype>
    __global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
    // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    //        |x| - 0.5 / sigma / sigma    otherwise
        CUDA_KERNEL_LOOP(index, n) { //for loop
            Dtype val = in[index];
            Dtype abs_val = abs(val);
            if (abs_val < 1.0 / sigma2) {
                out[index] = 0.5 * val * val * sigma2;
            }
            else {
                out[index] = abs_val - 0.5 / sigma2;
            }
        }
    }
```
  反向过程中根据求导公式可以得到如下式子，Backward的过程也如下所示
$$\frac{\partial Loss}{\partial x} = w_{in}\*w_{out}\*\frac{\partial SmoothL1(x)}{\partial x}$$

cpu版本可以自己实现，只需要把$GPU\_data\_diff$换成$cpu$,以及$gpu$的$smoothL1$写成$CPU$的即可。
```cpp
    //backward过程，根据导函数
    // f'()
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
        int count = diff_.count();

        // 反向即公式的smoothL1的偏导
        SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS >>>(
            count, diff_.gpu_data(), diff_.mutable_gpu_data(), sigma2_);
        CUDA_POST_KERNEL_CHECK;

        //此处的循环loop如同欧式损失函数，因为无法确认bottom[0]和bottom[1]，fc和label
        //的顺序，forward默认是0-1，因此如果0为label，则sign = -1;
        for (size_t i = 0; i < 2; i++) {
            if (propagate_down[i]) {
                const Dtype sign = (i == 0) ? 1:-1;//代码默许了label为bottom[1]
                //sign* loss/n;
                const Dtype alpha = sign*top_diff->gpu_diff()[0]/bottom[i]->num();
                //smoothL1输入的是diff_.gpu_data()
                caffe_cpu_axpby(
                    count,
                    alpha,
                    diff_.gpu_data(),//此处的data已经是SmoothL1返回的导数了
                    Dtype(0),
                    bottom[i]->mutable_gpu_diff()
                );
                if (has_weights_) {
                    caffe_gpu_mul(
                        count,
                        bottom[2]->gpu_data(),
                        bottom[i]->gpu_diff(),
                        bottom[i]->mutable_gpu_diff()
                    ); 乘以了内层的weight
                    caffe_gpu_mul(
                        count,
                        bottom[3]->gpu_data(),
                        bottom[i]->gpu_diff(),
                        bottom[i]->mutable_gpu_diff()
                    ); 乘以了外层的weight
                }
            }
        }
    }

    template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
  //       = sign(x)                   otherwise
        CUDA_KERNEL_LOOP(index, n) {
            Dtype val = in[index];
            Dtype abs_val = abs(val);
            if (abs_val < 1.0 / sigma2) {
              out[index] = sigma2 * val;
        }
        else {
              out[index] = (Dtype(0) < val) - (val < Dtype(0));//1或者-1
          }
        }
    }

```
cpu版本的SmoothL1前向和后向实现如下,cpu版本速度过慢，不建议使用
```cpp
    //前向 替换前向GPU中一部分
    const Dtype* in = diff_.cpu_data();
    Dtype* out = errors_.mutable_cpu_data();
    for (size_t i = 0; i < diff_.count(); i++) {
       Dtype val = in[index];
       Dtype abs_val = abs(val);
       if(abs_val < 1.0 / sigma2_){
           out[index] = 0.5 * val * val * sigma2_;
       }
       else{
           out[index] = abs_val - 0.5 / sigma2_;
       }
   }

   //反向，替换反向GPU的一部分
   const Dtype* in = diff_.cpu_data();
   Dtype* out = diff_.mutable_cpu_data();
   for (size_t i = 0; i < diff_.count(); i++) {
      Dtype val = in[index];
      Dtype abs_val = abs(val);
      if(abs_val < 1.0 / sigma2_){
          out[index] = sigma2_ *  val;
      }
      else{
          out[index] = (Dtype(0) < val) - (val < Dtype(0));
      }
   }

// smoothL1在目标检测的时候效果良好，由于多损失函数以及回归点的变换，bottom[2]和
// bottom[3]基本都存在，由于其函数特性，对偏远的点不敏感，因此可以替换L2loss
```
##### (4) SigmoidCrossEntropyLoss (交叉熵)
  交叉熵应用广泛，常作为二分类的损失函数，在$logistic$中使用，由于$sigmoid$的函数的输出特性，能够很好的以输出值代表类别概率。具体的公式如下所示:

  $$loss =  -\frac{1}{n}\sum_{1}^{n}(\hat{p_i}\*log(p_i)+(1-\hat{p_i})\*log(1-p_i)))$$
$$p_i = \frac{1}{1.+exp(-x_i)}$$
$$ \frac{\partial loss}{\partial x_i} = -\frac{1}{n}\*\sum_{i=1}^{n}((\hat{p_i}\*\frac{1}{p_i}\*p_i\*(1-p_i)-(1-\hat{p_i})\*\frac{1}{1-p_i}\*(1-p_i)\*p_i)) $$
$$= -\frac{1}{n}\sum_{i=1}^{n}(\hat{p_i}-p_i)$$
1.基本的数据成员
```cpp
    shared_ptr<SigmoidLayer<Dtype>>sigmoid_layer_;//layer参数
    shared_ptr<Blob<Dtype> > sigmoid_output_; // sigmoid输出的值N*C C一般==1
    shared_ptr<Blob<Dtype>* > sigmoid_bottom_vec_;// sigmoid函数的输入x
    shared_ptr<Blob<Dtype>* > sigmoid_top_vec_;// sigmoid函数的输出
```

2.基本的成员函数
    基本的成员函数为LayerSetup，Reshape ,Forward和Backward,实现如下:
```cpp
    //构建layer 中间有sigmoid函数过度，所以如同softmaxLoss类似过程
    template <typename Dtype>
    void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetup(
        const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
        LossLayer<Dtype>::LayerSetup(bottom,top);
        sigmoid_bottom_vec_.clear();
        sigmoid_bottom_vec_.push_back(bottom[0]);
        sigmoid_top_vec_.clear();
        sigmoid_top_vec_.push_back(sigmoid_output_.get());//sigmoid的输出
        sigmoid_layer_->Setup(sigmoid_bottom_vec_,sigmoid_top_vec_);
    }

    //Reshape函数 比较简单
    template <typename Dtype>
    void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
        LossLayer<Dtype>::Reshape(bottom,top);//步骤1
        sigmoid_layer_->Reshape(sigmoid_bottom_vec_,sigmoid_top_vec_);//步骤2
    }
```
这里Caffe实现的前向计算代码与公式有差异，具体原因如下
$\qquad  \hat{p}\*log(p)+(1-\hat{p})\*log(1-p) \\
\qquad \,= \hat{p}\*log(\frac{1}{1+e^{-x}})+(1-\hat{p})\*log(\frac{e^{-x}}{1+e^{-x}}) \\
\qquad =\hat{p}\*log(\frac{1}{1+e^{-x}})-\hat{p}\*log(\frac{e^{-x}}{1+e^{-x}})+log(\frac{e^{-x}}{1+e^{-x}}) \\
\qquad =\hat{p}\*x+log(\frac{e^{-x}}{1+e^{-x}})$

当$e^{-x}很大时, \frac{e^{-x}}{1+e^{-x}}$ 计算不准确，因此采用下种计算方式,当 $x<0$时,分子分母同时乘以$e^{x}$,有:

$$
\frac{e^{-x}}{1+e^{-x}}=
\begin{cases}
\  \frac{e^{-x}}{1+e^{-x}} \qquad x\ge0
\\
\  \frac{1}{1+e^{x}} \qquad \,\,\, x<0
\end{cases}
$$

从而得到:
$$
\hat{p}\*x+log(\frac{e^{-x}}{1+e^{-x}})=
\begin{cases}
\ \hat{p}\*x+log(\frac{e^{-x}}{1+e^{-x}}) = (\hat{p}-1)
\*x-log(1+e^{-x}) \quad x\ge0
\\
\ \hat{p}\*x+log(\frac{e^{-x}}{1+e^{-x}})=\hat{p}\*x-log(1+e^{x}) \quad\quad \qquad x<0
\end{cases}
$$

```cpp
    // Forward_cpu 前向函数，分布保存临时值，得到loss
    template <typename Dtype>
    void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*> & bottom,const vector<Blob<Dtype>*>& top){
        sigmoid_bottom_vec_[0] = bottom[0];//这一步多余，setup时已经保持一致了
        sigmoid_layer_->Forward_cpu(sigmoid_bottom_vec_,sigmoid_top_vec_);//Sigmoid
        const int count = bottom[0]->count();//N*1*1*1，输出一个概率值为预测1的
        const int num = bottom[0]->num();
        const Dtype* input_data = bottom[0]->cpu_data();
        const Dtype* target = bottom[1]->cpu_data();//真实label
        Dtype loss = 0;
        for (size_t i = 0; i < count; i++) {//遍历mini_batch
            loss -= input_data[i]*(target[i]-(input_data[i]>=0))-
                    log(1.+exp(input_data[i]-2*(input_data[i]>=0)));
        }
        top[0]->mutable_cpu_data()[0] = loss/num;//mini_batch
    }


    //backward的反向更新比较简单，-(target-predict)
    template <typename Dtype>
    void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
        const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*> & bottom){
        if (propagate_down[1]) {
            //label 不需要更新
        }
        if (propagate_down[0]) {
            const int count = bottom[0]->count();//N*1*1*1
            const int num = bottom[0]->num();// N
            const Dtype* sigmoid_output_data = sigmoid_output_.cpu_data();//预测值
            const Dtype* target = bottom[1]->cpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            // bottom_diff = predict - target_label
            caffe_sub(count,sigmoid_output_data,target,bottom_diff);
            const Dtype loss_weight = top[0]->cpu_diff()[0];
            //bottom_diff = bottom_diff*loss_weight/n
            caffe_scal(count,loss_weight/num,bottom_diff);
        }
    }
```

##### (5) CenterLoss (ECCV2016)
  ECCV2016年提出的新loss，让softmax能够训练出更好的内聚性的特征，思路比较简单，在SoftmaxLoss的基础上，添加了一个新的loss，Loss的表达式:
  $$\zeta_C = \frac{1}{2}\*\sum_{i=1}^{n}||x_i-c_{yi}||_2^2$$
  思路比较好理解，增加一个loss衡量样本特征与该类类心的距离，更新的公式如下:
  $$\frac{\partial \zeta_c}{\partial x_i} = x_i - c_{yi} \\
  \triangle c_j = \frac{\sum_{i=1}^{n}\delta{(y_i=j)}\*(c_j-x_i)}{1+\sum_{i=1}^{n}\delta{(y_i=j)}}$$
   $$c_j^{t+1} = c_j^t-\alpha\*\triangle{c_j^t}$$
  第二步骤类心特征更新仅仅更新当前样本所属的类别，分母加1为了防止分母为0，因此和softmax整合后整体的Loss如下所示：
  $$\zeta = \zeta_S+\lambda \zeta_C$$

1.基本数据成员
```cpp
    //基本数据用以保存center_Loss的layer params
    int N_;// 对应params的num_output,分类类别
    int K_;// 对应fc层的输出特征,
    int M_;// 对应于batch_size
    Blob<Dtype> distance_;//样本与类心的距离，distance为x - x_center重点
    Blob<Dtype> variation_sum_;// distance的负数， x_center- x
    Blob<Dtype> count_; // 类心的个数
    string distance_type_; // 距离的衡量 默认L2
```

2.基本的成员函数
  与一般的Loss层一样，有LayerSetup,Reshape,Forward,Backward,具体实现如下
```cpp
    // layersetup过程，center是N个中心，每个类心feature长度K
    template <typename Dtype>
    void CenterLossLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        CenterLossParameter loss_param = this->layer_param_.center_loss_param();
        N_ = loss_pram.num_output();//分类的类别，类心的个数,prototxt内设置
        distance_type_ = loss_pram.distance_type();
        const int axis = bottom[0]->CanonicalAxisIndex(loss_pram.axis());
        K_ = bottom[0].count(axis);//axis 默认为1，K_= fc*1*1,特征的长度
        M_ = bottom[0]->num(); // batch_size的大小
        if (this->blobs_.size() > 0) {
            //层内无参数.
        }
        else{
            this->blobs_.resize(1);//这里放center，各个类别的fc中心
            vector<int> center_shape(2);
            center_shape[0] = N_;
            center_shape[1] = K_;
            // 代表中心是N个中心，每个中心的feature长度为K_
            this.blobs_[0].resize(new Blob<Dtype>(center_shape));
            // 初始中心的填充方式
            shared_ptr<Filler<Dtype>>center_filler(GetFiller<Dtype>(
                loss_param.center_filler()));
            )
            center_filler->Fill(this->blobs_[0].get());
        }
        this->param_propagate_down_.resize(this->blobs_.size(),true);//类心也更新
    }

    // Reshape函数
    template <typename Dtype>
    void CenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
        const vector<Blob<Dtype>*>& top){
        LossLayer<Dtype>::Reshape(bottom,top);
        distance_.ReshapeLike(*bottom[0]);//bottom长度为N_*K_
        variation_sum_.ReshapeLike(*this->blobs_[0]);//一样的N_*K_
        vector<int>count_reshape(1);
        count_reshape[0]= N_;
        count_.Reshape(count_reshape);//N_类心的个数
    }

    //Forward_cpu ，得到loss
    // N_类别数，K_特征长度,M_mini_batch的样本个数
    template <typename Dtype>
    void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();//N_*K_
        const Dtype* label = bottom[1]->cpu_data();//N_*1;
        const Dtype* center = this->blobs_[0]->cpu_data();//N_K_
        Dtype* distance_data = distance_.mutable_cpu_data();//
        // i-t样本的距离
        for (size_t i = 0; i < M_; i++) {
            const int label_value = static_cast<int>(label[i]);//真是的样本类别
            //对应特征相减，用fc特征减去该类的类心，保存在distance_data上
            caffe_sub(K_,bottom+i*K_,center+label_value*K_,distance_data+i*K_);
        }
        Dtype dot;
        Dtype loss;
        if (distance_type_ == "L1") { //L1 loss,distance_ sum即可
            // 也可以写caffe_cpu_asum(M_*K_,distance_data);
            dot = caffe_cpu_asum(M_*K_,distance_.cpu_data());
            loss = dot/M_;
        }
        //L2,loss,distance_data*distance_data,然后M_样本sum
        else if(distance_type_ == "L2"){
            dot = caffe_cpu_dot(M_*K_,distance_.cpu_data(),distance_.cpu_data());
            loss = dot/M_/Dtype(2);
        }
        else{
            //不支持其他的距离衡量
        }
        top[0]->mutable_cpu_data()[0] = loss;
    }

    // Backward_cpu,更新data和center，
    template <typename Dtype>
    void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
        if (this->param_propagate_down_[0]) {//表示更新类心
            const Dtype* label = bottom[1]->cpu_data();
            Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
            Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
            int* count_data = count_.mutable_cpu_data();
            const Dtype* distance_data = distance_.cpu_data();//fc_center-fc_pre
            if (distance_type_ == "L1") {
                caffe_cpu_sign(M_*K_,distance_data,distance_.mutable_cpu_data());
            }
            caffe_set(N_*K_,Dtype(0),variation_sum_.mutable_cpu_data());
            caffe.set(N_,0,count_.mutable_cpu_data());//统计每个类别的个数

            for (size_t i = 0; i < M_; i++) {//样本循环
                const int label_value = static_cast<int>(label[i]);
                //variation_sum_data 初始为0，distance保存的即使x_i-x_center
                caffe_sub(K_,variation_sum_data+label_value*K_,
                    distance_data+i*K,variation_sum_data+label_value*K);
                count_data[label_value]++:
            }
            for (size_t i = 0; i < M_; i++) {
                const int label_value = static_cast<int>(label[0]);
                //1/(count+1)*(x_center-x_i)
                caffe_cpu_axpby(K_,Dtype(1)/(count_data[label_value]+1),
                variation_sum_data+label_value*K,1.,center_diff+label_value*K_);
            }
        }

        //类心更新完成后,跟新x
        if (propagate_down[0]) {//更新输入x
            //loss * 1/M * (x - x_center)
            caffe_copy(M_*K_,distance.cpu_data(),bottom[0]->mutable_cpu_diff());
            cafe_scal(M_*K_,top[0]->cpu_diff()[0]/M_,
            bottom[0]->mutable_cpu_diff());
        }
        if (propagate_down[1]) {
            // label不更新
        }
    }
```
 $CenterLoss$在多分类上较$Softmax$有提高，$loss \_weight$的设置可以确定$center \_loss$和$softmaxloss$的比重，能够很有效的使得网络能够最小化类内距离，加大区分度。





>本文作者： 张峰
>本文链接：[https://zhanglaplace.github.io/2017/10/20]( https://zhanglaplace.github.io/2017/10/20/Caffe%20Loss%E5%88%86%E6%9E%90/)
>版权声明：本博客所有文章，均采用CC BY-NC-SA 3.0 许可协议。转载请注明出处！

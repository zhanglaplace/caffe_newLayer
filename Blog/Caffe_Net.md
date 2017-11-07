---
title: Caffe Net分析
date: 2017-10-19 14:42:00
tags: [Caffe,DeepLearning]
categories: Caffe
---

# Caffe_Net

### 1.基本数据
```cpp
    vector<shared_ptr<Layer<Dtype> > > layers_; // 记录每一层的layer参数
    vector<vector<Blob<Dtype>*> > bottom_vecs_;
    vector<vector<int> > bottom_id_vecs_;
    vector<vector<bool> > bottom_need_backward_;
    /// top_vecs stores the vectors containing the output for each layer
    vector<vector<Blob<Dtype>*> > top_vecs_;
    vector<vector<int> > top_id_vecs_;
    vector<vector<int> > param_id_vecs_;
    vector<string> layer_names_;
    //learnable_params_[learnable_param_ids_[i]] == params_[i].get()
    vector<Blob<Dtype>*> learnable_params_;//层间权重与bias
```
<!--more-->
### 2. 常用的函数
    介绍了Caffe内的Net的常用函数:
```cpp
   const string& name(){return name_;}//网络的名称
   const vector<string>& layer_names{return layer_names_;}// net每层的layer名称
   // net内每层的layer的Blob名称
   const vector<string>& blob_names(){return blob_names_;}
   //net内层次间的权值与bias
   const vector<shared_ptr<Blob<Dtype>>>& blobs(){return blob_;};
   //net内的layers
   const vector<shared_ptr<Layer<Dtype>>>& layers(){return layers_;};
    //net->bottom_vecs() 返回该layer的输入，输出向量，
    //以及具体的 top_id_vecs[layer_id][top_id];
   const vector<vector<Blob<Dtype>*> >& bottom_vecs(){ return bottom_vecs_;}
   const vector<vector<Blob<Dtype>*> >& top_vecs() { return top_vecs_;}
   const vector<vector<int> >& bottom_id_vecs(){ return bottom_id_vecs_;}
   const vector<vector<int> >& top_id_vecs() { return top_id_vecs_;}
   void CopyTrainedLayersFrom(const string trained_filename);//加载权重
   //网络的输入输出
   //感觉等效于bottom_vecs_[0]
   const vector<Blob<Dtype>*>& input_blobs(){return net_input_blobs_;}
   const vector<Blob<Dtype>*>& output_blobs()
   {return net_output_blobs;}//top_vecs[top_vecs.size()-1];

   const int num_input(){return net_input_blobs_.size()};//输入blob的size
   //has_blob()然后find return
   const shared_ptr<Blob<Dtype>>blob_by_name(const string& blob_name);

  // 前向计算loss和网络的输出
  const vector<Blob<Dtype>*>& forward(Dtype* loss = NULL);
  // --- *loss = ForwardFromTo(0.layers_.size()-1);
  // --- 此处调用 Dtype* Net<Dtype>::ForwardFrom(int start,int end)
  for (size_t i = start; i < end; i++){
      //多态，调用具体的Layer的Forward函数,并返回该层次的loss
      Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i],top_vecs_[i]);
      loss += layer_loss;
  }
  return loss;

  // backward反向，更新权值
  void Net<Dtype>::Backward(){ //
      BackwardFromTo(layers_size()-1,0); // 具体函数实现如第三部分
      if (debug_info_) {
          /*层次的参数*/
      }
  }


```

### 3.具体函数实现
```cpp
template <typename Dtype>
  const int Net<Dtype>::AppendBottom(const NetParamter& param, int layer_id,
int bottom_id,set<string>* availabel_blobs,map<string,int>* blob_name_to_idx){
    const LayerParammeter& layer_param = param.layer(layer_id);
    const string& blob_name = layer_param.bottom(bottom_id);
    const int blob_id = (*blob_name_to_idx)[blob_name];
    //layer输入的shape等
    bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
    bottom_id_vecs_[layer_id].push_back(blob_id);
    //LOG CONV<--data 等,只要是丢入输入
  }

  // learnable_params_
  //conv的shape一般为num_output*input_channels*kernel_width*kernel_height
  //bias的shape一般为Num_output
  template <typename Dtype>
  void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                           const int param_id) {
       const int learnable_param_id = learnable_params_.size();
       learnable_params_.push_back(params_[net_param_id].get());
       learnable_param_ids_.push_back(learnable_param_id);
       has_params_lr_.push_back(param_spec->has_lr_mult());
       has_params_decay_.push_back(param_spec->has_decay_mult());
       params_lr_.push_back(param_spec->lr_mult());
       params_weight_decay_.push_back(param_spec->decay_mult());
  }

  template <typename Dtype>
  void Net<Dtype>::BackwardFromTo(int start,int end){
      for(int i = start;i >= end;--i){
         //backward 调用各层次的backward更新权值和bias
         layers_[i].Backward(top_vecs_[i],bottom_need_backward_[i],
                            bottom_vecs_[i]);
      }
  }



```


### 4.基本流程
  基本流程：Net构造函数开始

```cpp
  // 递归更新变量
  vectot<string>*stage ;
  int level;

  //起始调用
  net_.reset(new Net<float>(model_file, TEST));

  //送入prototxt文件和Test OR Train
  explicit Net(const string& param_file,Phase phase,const int level = 0,
      vector<string>* stage = NULL,const Net* root_net = NULL);

  // 解析保存在NetParamter param内,这里用到了
  //protobuf::TextFormat::Parse(FileInputStream*,param)
  ReadNetParamsFromTextFileOrDie(param_file,&param);

  // 读取了NetParamter 后需要进行整个网络的初始化工作
  Init(param); //初始化网络的接口，下续为具体实现
  FilterNet(param, &filtered_param);// 打印网络结构

  /*内部会完成split added 如果有必要(残差结构),记录层与层之间的联系关系与层次的名称
  等，是否有loss_weight，layer的size等*/
  InsertSplits(filtered_param,&param);

  for (size_t i = 0; i < param.layer_size(); i++) { //遍历setupLayer
     const LayerParammeter& layer_param = param.layer(i);//层次的参数
     layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
     // CreateLayer会走layer_factory的CreateLayer的注册 ,比如input,conv,bn...
     layer_names_.push_back(layer_param.name());

     //开始继续遍历每层输入的具体的细节,第i个layer的第botom_id个输入
     for (size_t bottom_id = 0; bottom_id < layer_param.bottom_size();
     bottom_id++) {
        const int blob_id =
        AppendBottom(param,i,bottom_id,&availabel_blobs,&blob_name_to_idx);
     }

     //开始继续遍历每层输出的具体细节，第i个layer的第 top_id的输出
     for (size_t top_id = 0; top_id < layer_param.top_size();
     top_id++) {
        AppendTop(param,i,top_id,&availabel_blobs,&blob_name_to_idx);
         if (layer_param.type()== "Input") {//输入
           const int blob_id = blobs_.size() - 1;
           net_input_blob_indices_.push_back(blob_id);
           net_input_blobs_.push_back(blobs_[blob_id].get());
         }
     }

   //多态，具体调用具体的layer的Setup函数
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);

    //每个输出遍历
    for (size_t top_id = 0; top_id < top_vecs_[layer_id].size();
    top_id++) {
       /*完成层次的blob_loss_weights,并统计memory_used_*/;
       memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    //总的memory_used_: memory_used_*sizeof(Dtype);

    //如果层次间有学习权值和偏置，则需要再次设置，比如conv
    //num_param_blobs weights And bias
    // relu pooling等层无中间权值参数，则num_param_blobs = 0
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
        AppendParam(param, layer_id, param_id);
    }

  }

  /*接下来需要研究网络的backwards问题，决定哪些层次对loss有贡献，并且检查哪些层次
  不需要back_propagate_down操作，遍历是反向的操作一个layer是否需要回溯计算，主要
  依据两个方面：(1)该layer的top blob 是否参与loss的计算；(2):该layer的bottom
  blob 是否需要回溯计算，比如Data层一般就不需要backward computation */
  for (size_t layer_id = layers_.size()-1; layer_id >= 0; --layer_id){
     bool layer_contributes_loss = false;//默认是无贡献的
     bool layer_skip_propagate_down = true;// 默认不参与backwards的loss贡献

    //Layer内的输出遍历
    for (size_t top_id = 0; top_id < top_vecs_[layer_id].size();
    top_id++) {
        //blob_name_[index]名字
       string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
       if (layer_[layer_id]->loss(top_id)||
       blobs_under_loss.find(blob_name) != blobs_under_loss.end()) {
           //该层次的layerloss不为0或者loss_weight = 1;
          layer_contributes_loss = true;
       }
       if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
          layer_skip_propagate_down = false;
       }
    }

    //同理 Layer内的输入遍历
    for (size_t bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
    bottom_id++) {
      if (layer_contributes_loss) {
        string* blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      }
      else{
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        string&blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blok_skip_backp.insert(blob_name);
      }
    }

    /*code*/

  }//init函数尾

```

### 5.说明
   blob_name_to_idx是一个局部变量，其实它是在当前layer的top blob 和下一层的bottom blob间起着一个桥梁作用。 blob_name_to_idx中元素的pair是从网络最开始一层一层搭建的过程中压入map的，其中的name和id都是不重复的。name是关键字，不重复是map数据结构的必然要求，id也是不重复的，—0,1,2...blob_name_to_idx和blobs_一样，在"Normal output"的情形下，每次遍历到一个top blob的时候都会更新。

>本文作者： 张峰
>本文链接： https://zhanglaplace.github.io/2017/10/19/Caffe_Net/
>版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！

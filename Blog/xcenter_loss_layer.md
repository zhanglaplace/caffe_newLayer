
```cpp

#ifndef CAFFE_XCENTER_LOSS_LAYER_HPP_
#define CAFFE_XCENTER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	class XCenterLossLayer : public LossLayer<Dtype> {
	public:
		explicit XCenterLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "XCenterLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return -1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int M_;
		int K_;
		int N_;
		int lambda_;
		Blob<Dtype> distance_;
		Blob<Dtype> variation_sum_;
		Blob<Dtype> similarity_sum_;
    vector<Dtype> center_module_;
	};

}  // namespace caffe

#endif  // CAFFE_XCENTER_LOSS_LAYER_HPP_
```


```cpp
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/xcenter_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void XCenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
const XCenterLossParameter& param = this->layer_param_.x_center_loss_param();
const int num_output = param.num_output();
lambda_ = param.lambda();
N_ = num_output;
const int axis = bottom[0]->CanonicalAxisIndex(
  param.axis());
K_ = bottom[0]->count(axis);
// Check if we need to set up the weights
if (this->blobs_.size() > 0) {
  LOG(INFO) << "Skipping parameter initialization";
}
else {
  this->blobs_.resize(1);
  // Intialize the weight
  vector<int> center_shape(2);
  center_shape[0] = N_;
  center_shape[1] = K_;
  this->blobs_[0].reset(new Blob<Dtype>(center_shape));
  // fill the weights
  shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
    param.center_filler()));
  center_filler->Fill(this->blobs_[0].get());

}  // parameter initialization
this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void XCenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
CHECK_EQ(bottom[1]->channels(), 1);
CHECK_EQ(bottom[1]->height(), 1);
CHECK_EQ(bottom[1]->width(), 1);
M_ = bottom[0]->num();
// The top shape will be the bottom shape with the flattened axes dropped,
// and replaced by a single axis with dimension num_output (N_).
LossLayer<Dtype>::Reshape(bottom, top);
distance_.ReshapeLike(*bottom[0]);
variation_sum_.ReshapeLike(*this->blobs_[0]);
long int similarity_count = N_*(N_-1) / 2;
vector<int>similarity_shape(this->blobs_[0]->shape());
similarity_shape[0] = similarity_count;
similarity_sum_.Reshape(similarity_shape);
center_module_.resize(N_);
}

template <typename Dtype>
void XCenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
const Dtype* bottom_data = bottom[0]->cpu_data();
const Dtype* label = bottom[1]->cpu_data();
const Dtype* center = this->blobs_[0]->cpu_data();
Dtype* distance_data = distance_.mutable_cpu_data();


// center_loss
// the i-th distance_data
for (int i = 0; i < M_; i++) {
  const int label_value = static_cast<int>(label[i]);
  // D(i,:) = X(i,:) - C(y(i),:)
  caffe_sub(K_, bottom_data + i * K_, center + label_value * K_, distance_data + i * K_);
}
Dtype dot = caffe_cpu_dot(M_ * K_, distance_.cpu_data(), distance_.cpu_data());


// x_center_loss

Dtype* similarity_data = similarity_sum_.mutable_cpu_data();

vector<float>similarity_py(N_*(N_-1)/2,0.0);
for (int i = 0; i < N_;++i){ //类别遍历
  //求得各个类别的center模长
  center_module_[i]  = sqrt(caffe_cpu_dot(K_,center_+i*K,center_+i*K_));
}
for (size_t i = 0; i < N_-1; i++) {
  for (size_t j = i+1; j < N_; j++) {
     similarity_py
  }
}


Dtype dot1 = Dtype(0);
for (int n = 0; n < similarity_py.size();++n){
  float sum = 0;
  for (int k = 0; k < K_; ++k){
    sum += similarity_data[n*K_ + k];
  }
  dot1 += sum / similarity_py[n]+ 1;
}
dot1 = Dtype(2)*lambda_* dot1 /similarity_py.size() ;
Dtype loss = loss / M_ / Dtype(2);
top[0]->mutable_cpu_data()[0] = loss + dot1;
}

template <typename Dtype>
void XCenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down,
const vector<Blob<Dtype>*>& bottom) {
// Gradient with respect to centers
if (this->param_propagate_down_[0]) {
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
  const Dtype* distance_data = distance_.cpu_data();

  // \sum_{y_i==j}
  caffe_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());
  for (int n = 0; n < N_; n++) {
    int count = 0;
    for (int m = 0; m < M_; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == n) {
        count++;
        caffe_sub(K_, variation_sum_data + n * K_, distance_data + m * K_, variation_sum_data + n * K_);
      }
    }
    caffe_axpy(K_, (Dtype)1. / (count + (Dtype)1.), variation_sum_data + n * K_, center_diff + n * K_);
  }
}
// Gradient with respect to bottom data
if (propagate_down[0]) {
  caffe_copy(M_ * K_, distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
  caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_cpu_diff());
}
if (propagate_down[1]) {
  LOG(FATAL) << this->type()
    << " Layer cannot backpropagate to label inputs.";
}
}

#ifdef CPU_ONLY
STUB_GPU(XCenterLossLayer);
#endif

INSTANTIATE_CLASS(XCenterLossLayer);
REGISTER_LAYER_CLASS(XCenterLoss);

}  // namespace caffe
```

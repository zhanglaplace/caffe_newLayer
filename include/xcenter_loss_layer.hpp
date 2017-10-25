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
		vector<vector<Dtype> > center_dot_;
	};

}  // namespace caffe

#endif  // CAFFE_XCENTER_LOSS_LAYER_HPP_
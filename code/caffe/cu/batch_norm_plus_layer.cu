#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_plus_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>
namespace caffe {

	template <typename Dtype>
	__global__ void BatchNormPlusForward(const int n, Dtype* in,const Dtype* scale,
		const Dtype* bias, const int scale_dim, const int inner_dim,
		Dtype* out) {
		CUDA_KERNEL_LOOP(index, n) {
			const int scale_index = (index / inner_dim) % scale_dim;
			out[index] = in[index] * scale[scale_index] + bias[scale_index];
		}
	}

	template <typename Dtype>
	void BatchNormPlusLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		int num = bottom[0]->shape(0);
		int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));

		if (bottom[0] != top[0]) {
			caffe_copy(bottom[0]->count(), bottom_data, top_data);
		}


		if (use_global_stats_) {
			// use the stored mean/variance estimates.
			const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
				0 : 1 / this->blobs_[2]->cpu_data()[0];
			caffe_gpu_scale(variance_.count(), scale_factor,
				this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
			caffe_gpu_scale(variance_.count(), scale_factor,
				this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
		}
		else {
			// compute mean
			caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
				1. / (num * spatial_dim), bottom_data,
				spatial_sum_multiplier_.gpu_data(), 0.,
				num_by_chans_.mutable_gpu_data());
			caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
				num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
				mean_.mutable_gpu_data());
		}

		// subtract mean
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
			num_by_chans_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, -1, num_by_chans_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), 1., top_data);

		if (!use_global_stats_) {
			// compute variance using var(X) = E((X-EX)^2)
			caffe_gpu_powx(top[0]->count(), top_data, Dtype(2),
				temp_.mutable_gpu_data());  // (X-EX)^2
			caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
				1. / (num * spatial_dim), temp_.gpu_data(),
				spatial_sum_multiplier_.gpu_data(), 0.,
				num_by_chans_.mutable_gpu_data());
			caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
				num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
				variance_.mutable_gpu_data());  // E((X_EX)^2)

			// compute and save moving average
			this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
			this->blobs_[2]->mutable_cpu_data()[0] += 1;
			caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
				moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
			int m = bottom[0]->count() / channels_;
			Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
			caffe_gpu_axpby(variance_.count(), bias_correction_factor,
				variance_.gpu_data(), moving_average_fraction_,
				this->blobs_[1]->mutable_gpu_data());
		}

		// normalize variance
		caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
		caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
			variance_.mutable_gpu_data());

		// replicate variance to input size
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,
			num_by_chans_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, 1., num_by_chans_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
		caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
		// TODO(cdoersch): The caching is only needed because later in-place layers
		//                 might clobber the data.  Can we skip this if they won't?
		if (!use_global_stats_) {
			caffe_copy(x_norm_.count(), top_data,
				x_norm_.mutable_gpu_data());
		}

		// add scale layer by zhangfeng
		bool scale_bias = this->layer_param_.batch_norm_plus_param().scale_bias();
		if (scale_bias){
            const Dtype* scale_data = this->blobs_[3]->gpu_data();
			const Dtype* bias_data = this->blobs_[4]->gpu_data();
			const int count = top[0]->count();
			BatchNormPlusForward<Dtype> << <CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS >> >(
				count, top_data,scale_data, bias_data, channels_, spatial_dim, top_data);
		}
		caffe_copy(x_temp_top.count(), top_data, x_temp_top.mutable_gpu_data());

	}

	template <typename Dtype>
	__global__ void BatchNormPlusBackward(const int n, const Dtype* in, const Dtype* scale,
		const int scale_dim, const int inner_dim,
		Dtype* out) {
		CUDA_KERNEL_LOOP(index, n) {
			const int scale_index = (index / inner_dim) % scale_dim;
			out[index] = in[index] * scale[scale_index];
		}
	}



	template <typename Dtype>
	void BatchNormPlusLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff;
		int num = bottom[0]->shape(0);
		int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);
		if (bottom[0] != top[0]) {
			top_diff = top[0]->gpu_diff();
		}
		else {
			caffe_copy(x_temp_top.count(), top[0]->gpu_diff(),
				x_temp_top.mutable_gpu_diff());
			top_diff = x_temp_top.gpu_diff();
		}


		//scale 
		bool scale_bias = this->layer_param_.batch_norm_plus_param().scale_bias();//
		if (scale_bias){//��Ҫ����alpha��beta
            //scale
			Dtype* scale_diff = this->blobs_[3]->mutable_gpu_diff();
			//1 dE/dy * x_norm
			caffe_gpu_mul<Dtype>(top[0]->count(), top_diff, x_norm_.gpu_data(), x_temp_top.mutable_gpu_data());

			// 2.��sum 
			caffe_gpu_gemv<Dtype>(CblasNoTrans, num*channels_, spatial_dim, 1.,
				x_temp_top.gpu_data(), spatial_sum_multiplier_.gpu_data(), 0.,
				num_by_chans_.mutable_gpu_data()); // NC* HW * HW(1)*1 = NC*1
			caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1,
				num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0, scale_diff);

			// bias DE/dy 
			Dtype* bias_diff = this->blobs_[4]->mutable_gpu_diff();
			caffe_gpu_gemv<Dtype>(CblasNoTrans, num*channels_, spatial_dim, 1.,
				top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
				num_by_chans_.mutable_gpu_data()); // NC* HW * HW(1)*1 = NC*1
			caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1,
				num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0, bias_diff);

			//����dE/dx= dE/dy * scale; ����sum
			const Dtype* scale_data = this->blobs_[3]->gpu_data();
			Dtype* x_norm_diff = x_norm_.mutable_gpu_diff();
			const int count = top[0]->count();
			BatchNormPlusBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
				<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, top_diff, scale_data, channels_, spatial_dim, x_norm_diff);
			/*
			for (int n = 0; n < num; n++){
				for (int c = 0; c < channels_; c++){
					Dtype factory = scale_data[c];
					caffe_gpu_scale(spatial_dim, factory, top_diff, x_norm_diff);
					top_diff += spatial_dim;
					x_norm_diff += spatial_dim;
				}
			}
			*/
			top_diff = x_norm_.gpu_diff();
		}


		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		if (use_global_stats_) {
			caffe_gpu_div(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
			return;
		}
		const Dtype* top_data = x_norm_.gpu_data();
		//int num = bottom[0]->shape()[0];
		//int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
		// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
		//
		// dE(Y)/dX =
		//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
		//     ./ sqrt(var(X) + eps)
		//
		// where \cdot and ./ are hadamard product and elementwise division,
		// respectively, dE/dY is the top diff, and mean/var/sum are all computed
		// along all dimensions except the channels dimension.  In the above
		// equation, the operations allow for expansion (i.e. broadcast) along all
		// dimensions except the channels dimension where required.

		// sum(dE/dY \cdot Y)
		caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
		caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
			bottom_diff, spatial_sum_multiplier_.gpu_data(), 0.,
			num_by_chans_.mutable_gpu_data());
		caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
			num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
			mean_.mutable_gpu_data());

		// reshape (broadcast) the above
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
			num_by_chans_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, 1., num_by_chans_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

		// sum(dE/dY \cdot Y) \cdot Y
		caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

		// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
		caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
			top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
			num_by_chans_.mutable_gpu_data());
		caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
			num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
			mean_.mutable_gpu_data());
		// reshape (broadcast) the above to make
		// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
			num_by_chans_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
			spatial_dim, 1, 1., num_by_chans_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

		// dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
		caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff,
			Dtype(-1. / (num * spatial_dim)), bottom_diff);

		// note: temp_ still contains sqrt(var(X)+eps), computed during the forward
		// pass.
		caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(BatchNormPlusLayer);


}  // namespace caffe

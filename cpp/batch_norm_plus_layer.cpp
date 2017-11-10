#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_plus_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

namespace caffe {
	template <typename Dtype>
	void BatchNormPlusLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		BatchNormPlusParameter param = this->layer_param_.batch_norm_plus_param();
		moving_average_fraction_ = param.moving_average_fraction();
		use_global_stats_ = this->phase_ == TEST;
		if (param.has_use_global_stats())
			use_global_stats_ = param.use_global_stats();
		if (bottom[0]->num_axes() == 1)
			channels_ = 1;
		else
			channels_ = bottom[0]->shape(1);
		eps_ = param.eps();
		if (this->blobs_.size() > 0) { //测试阶段
			LOG(INFO) << "Skipping parameter initialization";
		}
		else {
			this->blobs_.resize(3);
			vector<int> sz;
			sz.push_back(channels_);
			this->blobs_[0].reset(new Blob<Dtype>(sz));
			this->blobs_[1].reset(new Blob<Dtype>(sz));
			sz[0] = 1;
			this->blobs_[2].reset(new Blob<Dtype>(sz));
			for (int i = 0; i < 3; ++i) {
				caffe_set(this->blobs_[i]->count(), Dtype(0),
					this->blobs_[i]->mutable_cpu_data());
			}
			if (param.has_scale_bias()){ // 增加scale层的alpha和beta
				this->blobs_.resize(5);
				sz[0] = channels_;
				this->blobs_[3].reset(new Blob<Dtype>(sz)); // alpha C
				this->blobs_[4].reset(new Blob<Dtype>(sz)); // beta C

				if (param.has_scale_filler()){
					shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(param.scale_filler()));
					scale_filler->Fill(this->blobs_[3].get());
				}
				else{
					caffe_set(this->blobs_[3]->count(), Dtype(1), this->blobs_[3]->mutable_cpu_data());
				}

				if (param.has_bias_filler()){
					shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(param.bias_filler()));
					bias_filler->Fill(this->blobs_[4].get());
				}
				else{
					caffe_set(this->blobs_[4]->count(), Dtype(1), this->blobs_[4]->mutable_cpu_data());
				}
			}
		}
	}

	template <typename Dtype>
	void BatchNormPlusLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (bottom[0]->num_axes() >= 1)
			CHECK_EQ(bottom[0]->shape(1), channels_);
		top[0]->ReshapeLike(*bottom[0]);

		vector<int> sz;
		sz.push_back(channels_);
		mean_.Reshape(sz);
		variance_.Reshape(sz);
		temp_.ReshapeLike(*bottom[0]);
		x_norm_.ReshapeLike(*bottom[0]);
		x_temp_top.ReshapeLike(*bottom[0]);// add for temp top_data
		sz[0] = bottom[0]->shape(0);
		batch_sum_multiplier_.Reshape(sz);

		int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
		if (spatial_sum_multiplier_.num_axes() == 0 ||
			spatial_sum_multiplier_.shape(0) != spatial_dim) {
			sz[0] = spatial_dim;
			spatial_sum_multiplier_.Reshape(sz);
			Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
			caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
		}

		int numbychans = channels_*bottom[0]->shape(0);
		if (num_by_chans_.num_axes() == 0 ||
			num_by_chans_.shape(0) != numbychans) {
			sz[0] = numbychans;
			num_by_chans_.Reshape(sz);
			caffe_set(batch_sum_multiplier_.count(), Dtype(1),
				batch_sum_multiplier_.mutable_cpu_data());
		}
	}

	template <typename Dtype>
	void BatchNormPlusLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int num = bottom[0]->shape(0);
		int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);

		if (bottom[0] != top[0]) {
			caffe_copy(bottom[0]->count(), bottom_data, top_data);
		}

		if (use_global_stats_) {
			// use the stored mean/variance estimates.
			const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
				0 : 1 / this->blobs_[2]->cpu_data()[0];
			caffe_cpu_scale(variance_.count(), scale_factor,
				this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
			caffe_cpu_scale(variance_.count(), scale_factor,
				this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
		}
		else {
			// compute mean
			caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
				1. / (num * spatial_dim), bottom_data,
				spatial_sum_multiplier_.cpu_data(), 0.,
				num_by_chans_.mutable_cpu_data());
			caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
				num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
				mean_.mutable_cpu_data());
		}

		// subtract mean
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, -1, num_by_chans_.cpu_data(),
			spatial_sum_multiplier_.cpu_data(), 1., top_data);

		if (!use_global_stats_) {
			// compute variance using var(X) = E((X-EX)^2)
			caffe_powx(top[0]->count(), top_data, Dtype(2),
				temp_.mutable_cpu_data());  // (X-EX)^2
			caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
				1. / (num * spatial_dim), temp_.cpu_data(),
				spatial_sum_multiplier_.cpu_data(), 0.,
				num_by_chans_.mutable_cpu_data());
			caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
				num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
				variance_.mutable_cpu_data());  // E((X_EX)^2)

			// compute and save moving average
			this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
			this->blobs_[2]->mutable_cpu_data()[0] += 1;
			caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
				moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
			int m = bottom[0]->count() / channels_;
			Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
			caffe_cpu_axpby(variance_.count(), bias_correction_factor,
				variance_.cpu_data(), moving_average_fraction_,
				this->blobs_[1]->mutable_cpu_data());
		}

		// normalize variance
		caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
		caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
			variance_.mutable_cpu_data());

		// replicate variance to input size
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, 1., num_by_chans_.cpu_data(),
			spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
		caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
		// TODO(cdoersch): The caching is only needed because later in-place layers
		//                 might clobber the data.  Can we skip this if they won't?
		caffe_copy(x_norm_.count(), top_data,
			x_norm_.mutable_cpu_data());

		// 此处处理完成后需要做的就是后续的scale层需要做的了
		bool scale_bias = this->layer_param_.batch_norm_plus_param().scale_bias();//
		if (scale_bias){ //没有scale和bias 则不要scaleLayer
			//如果有scale_bias则需要做前向计算
			// alpha* x_norm + bias 
			const Dtype* scale_data = this->blobs_[3]->cpu_data();
			const Dtype* bias_data = this->blobs_[4]->cpu_data();
			for (int n = 0; n < num; n++){
				for (int  c = 0; c < channels_; c++){
					const Dtype factory = scale_data[c];
					const Dtype bias = bias_data[c];
					caffe_cpu_scale(spatial_dim, factory, top_data, top_data);
					caffe_add_scalar(spatial_dim, bias, top_data);
					top_data += spatial_dim;
				}
			}
		}
		caffe_copy(x_temp_top.count(), top_data,
			x_temp_top.mutable_cpu_data());
	}

	template <typename Dtype>
	void BatchNormPlusLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		bool scale_bias = this->layer_param_.batch_norm_plus_param().scale_bias();//
		const Dtype* top_diff;
		int num = bottom[0]->shape(0);
		int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_); 
		if (bottom[0] != top[0]) {
			top_diff = top[0]->cpu_diff();
		}
		else {
			caffe_copy(x_temp_top.count(), top[0]->cpu_diff(), x_temp_top.mutable_cpu_diff());
			top_diff = x_temp_top.cpu_diff();
		}
		if (scale_bias){//需要计算alpha和beta
			//scale
			Dtype* scale_diff = this->blobs_[3]->mutable_cpu_diff();
			//1 dE/dy * x_norm
			caffe_mul<Dtype>(top[0]->count(), top_diff, x_norm_.cpu_data(), x_temp_top.mutable_cpu_data());

			// 2.求sum 
			caffe_cpu_gemv<Dtype>(CblasNoTrans, num*channels_, spatial_dim, 1.,
				x_temp_top.cpu_data(), spatial_sum_multiplier_.cpu_data(), 0.,
				num_by_chans_.mutable_cpu_data()); // NC* HW * HW(1)*1 = NC*1
			caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1,
				num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0, scale_diff);

			// bias DE/dy 
			Dtype* bias_diff = this->blobs_[4]->mutable_cpu_diff();
			caffe_cpu_gemv<Dtype>(CblasNoTrans, num*channels_, spatial_dim, 1.,
				top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
				num_by_chans_.mutable_cpu_data()); // NC* HW * HW(1)*1 = NC*1
			caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1,
				num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0, bias_diff);

			//计算dE/dx= dE/dy * scale; 在求sum
			const Dtype* scale_data = this->blobs_[3]->cpu_data();
			Dtype* x_norm_diff = x_norm_.mutable_cpu_diff();
			for (int n = 0; n < num; n++){
				for (int c = 0; c < channels_; c++){
					Dtype factory = scale_data[c];
					caffe_cpu_scale(spatial_dim, factory, top_diff, x_norm_diff);
					top_diff += spatial_dim;
					x_norm_diff += spatial_dim;
				}
			}
			top_diff = x_norm_.cpu_diff();
		}
		
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (use_global_stats_) {
			caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
			return;
		}
		const Dtype* top_data = x_norm_.cpu_data();
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
		caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
		caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
			bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
			num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
			mean_.mutable_cpu_data());

		// reshape (broadcast) the above
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
			spatial_dim, 1, 1., num_by_chans_.cpu_data(),
			spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

		// sum(dE/dY \cdot Y) \cdot Y
		caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

		// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
		caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
			top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
			num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
			mean_.mutable_cpu_data());
		// reshape (broadcast) the above to make
		// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
			batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
			num_by_chans_.mutable_cpu_data());
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
			spatial_dim, 1, 1., num_by_chans_.cpu_data(),
			spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

		// dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
		caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
			Dtype(-1. / (num * spatial_dim)), bottom_diff);

		// note: temp_ still contains sqrt(var(X)+eps), computed during the forward
		// pass.
		caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
	}


#ifdef CPU_ONLY
	STUB_GPU(BatchNormPlusLayer);
#endif

	INSTANTIATE_CLASS(BatchNormPlusLayer);
	REGISTER_LAYER_CLASS(BatchNormPlus);
}  // namespace caffe

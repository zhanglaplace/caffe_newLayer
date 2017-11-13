#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/island_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void IslandLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const IslandLossParameter& param = this->layer_param_.island_loss_param();
		const int num_output = param.num_output();
		lambda_ = param.lambda();
		N_ = num_output;
		const int axis = bottom[0]->CanonicalAxisIndex(
			param.axis());
		// Dimensions starting from "axis" are "flattened" into a single
		// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
		// and axis == 1, N inner products with dimension CHW are performed.
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
	void IslandLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
		center_module_.resize(N_);//记录每个center的模
		center_dot_.resize(N_);// 记录c_i*c_j的值
		for (int i = 0; i < center_dot_.size();++i){
			center_dot_[i].resize(N_,0);
		}
	}

	template <typename Dtype>
	void IslandLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
		
		
		// xcenter_loss
		Dtype loss1 = Dtype(0);


		for (int i = 0; i < N_;++i){
			center_module_[i] = sqrt(caffe_cpu_dot(K_, center + i*K_, center + i*K_));
		}
		for (int i = 0; i < N_;++i){
			for (int j = 0; j < N_;++j){
				if (i == j ){
					continue;
				}
				else{
					center_dot_[i][j] = caffe_cpu_dot(K_, center + i*K_, center + j*K_);
					loss1 += center_dot_[i][j] / (center_module_[i] * center_module_[j])+1; 
				}
			}
		}
		

		/*
		Dtype* similarity_data = similarity_sum_.mutable_cpu_data();
		vector<float>similarity_py(N_*(N_-1)/2,0.0);
		for (int i = 0; i < N_-1;++i){ //类别遍历
			for (int j = i + 1; j < N_;++j){
				//center_i* center_j对应相乘
				caffe_mul(K_, center + i*K_, center + j*K_, similarity_data + (i*N_ + j)*K_);
				//||center_i||*||center_j||
				similarity_py[i*N_ + j] = sqrt(caffe_cpu_dot(K_, center + i*K_, center + i*K_))*sqrt(caffe_cpu_dot(K_, center + j*K_, center + j*K_));
			}
		}
		*/
		loss1  = lambda_* loss1 /N_/(N_-1) ;
		Dtype loss = loss / M_ / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss + loss1;
	}

	template <typename Dtype>
	void IslandLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		// Gradient with respect to centers
		if (this->param_propagate_down_[0]) {
			const Dtype* label = bottom[1]->cpu_data();
			Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
			const Dtype* center_data = this->blobs_[0]->cpu_data();
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


			//xcenter_loss backward
			for (int n = 0; n < N_; ++n){
				Dtype double_center_module_n = center_module_[n] * center_module_[n];
				for (int i = 0; i < N_; ++i){
					if (i == n){
						continue;
					}
					else{ // 更新i
						Dtype alpha = center_module_[n] * center_module_[i];
						Dtype belta = center_dot_[n][i] / (alpha*double_center_module_n);
						//alpha*c_i-beta*c_n
						for (int k = 0; k < K_; ++k){
							//由于重复计算，实际计算的次数为2因此 center_diff的值需要乘以2
							center_diff[n*K_ + k] = 2*lambda_/(N_-1)*(alpha*center_data[i*K_ + k] - belta*center_data[n*K_ + k]);
						}
					}
				}
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
	STUB_GPU(IslandLossLayer);
#endif

	INSTANTIATE_CLASS(IslandLossLayer);
	REGISTER_LAYER_CLASS(IslandLoss);

}  // namespace caffe
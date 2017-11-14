#include "stdafx.h"
#include "faceDetect.h"


/******************************************************
// ������:CompareBBox
// ˵��: bbox�÷�paixu
// ����: �ŷ�
// ʱ��: 2017.11.14
// ��ע:
/*******************************************************/
bool CompareBBox(const FaceInfos & a, const FaceInfos & b) {
	return a.bbox.score > b.bbox.score;
}


/******************************************************
// ������:IoU
// ˵��:�����������ε�IOU
// ����:�ŷ�
// ʱ��:2017.11.14
// ��ע:
/*******************************************************/
float MTCNN_ONET::IoU(float xmin, float ymin, float xmax, float ymax,
	float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom) {
	float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
	float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
	if (iw <= 0 || ih <= 0) // �������β��ཻ
		return 0;
	float s = iw*ih; // a ��b
	if (is_iom) {
		
		return s / min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));;
	}
	else {
		return s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
	}
}


/******************************************************
// ������:NMS
// ˵��: �Ǽ���ֵ����
// ����:�ŷ�
// ʱ��:2017.11.14
// ��ע:
/*******************************************************/
std::vector<FaceInfos> MTCNN_ONET::NMS(std::vector<FaceInfos>& bboxes,
	float thresh, char methodType) {
	std::vector<FaceInfos> bboxes_nms;
	if (bboxes.size() == 0) {
		return bboxes_nms;
	}
	std::sort(bboxes.begin(), bboxes.end(), CompareBBox); // ������������scores�÷ֽ�������

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}

		bboxes_nms.push_back(bboxes[select_idx]);
		mask_merged[select_idx] = 1;

		// ѡȡ��ǰ���÷ֵ�scoreΪĿ��
		FaceBox select_bbox = bboxes[select_idx].bbox;
		float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
		float x1 = static_cast<float>(select_bbox.xmin);
		float y1 = static_cast<float>(select_bbox.ymin);
		float x2 = static_cast<float>(select_bbox.xmax);
		float y2 = static_cast<float>(select_bbox.ymax);

		select_idx++;
#pragma omp parallel for num_threads(threads_num)
		for (int32_t i = select_idx; i < num_bbox; i++) { // ����������Ŀ��
			if (mask_merged[i] == 1)
				continue;

			FaceBox & bbox_i = bboxes[i].bbox;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
			float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
			float area_intersect = w * h;
			// ���IOU A��B = area_intersect, A = area1 ,B = area2
			switch (methodType) {
			case 'u': // 
				if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}



/******************************************************
// ������:BBoxRegression
// ˵��:BBox�ع�
// ����:�ŷ�
// ʱ��:2017.11.4
// ��ע:
/*******************************************************/
void MTCNN_ONET::BBoxRegression(vector<FaceInfos>& bboxes) {
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float *bbox_reg = bboxes[i].bbox_reg;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		bbox.xmin += bbox_reg[0] * w;
		bbox.ymin += bbox_reg[1] * h;
		bbox.xmax += bbox_reg[2] * w;
		bbox.ymax += bbox_reg[3] * h;
	}
}

/******************************************************
// ������:BBoxPad
// ˵��:BBox�߽��ж�
// ����:�ŷ�
// ʱ��:2017.11.4
// ��ע:��ֹͼƬԽ��
/*******************************************************/
void MTCNN_ONET::BBoxPad(vector<FaceInfos>& bboxes, int width, int height) {
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		bbox.xmin = round(max(bbox.xmin, 0.f));
		bbox.ymin = round(max(bbox.ymin, 0.f));
		bbox.xmax = round(min(bbox.xmax, width - 1.f));
		bbox.ymax = round(min(bbox.ymax, height - 1.f));
	}
}


/******************************************************
// ������:BBoxPadSquare
// ˵��:BBox�߽��ж�
// ����:�ŷ�
// ʱ��:2017.11.14
// ��ע:
/*******************************************************/
void MTCNN_ONET::BBoxPadSquare(vector<FaceInfos>& bboxes, int width, int height) {
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		float side = h > w ? h : w;
		bbox.xmin = round(max(bbox.xmin + (w - side)*0.5f, 0.f));

		bbox.ymin = round(max(bbox.ymin + (h - side)*0.5f, 0.f));
		bbox.xmax = round(min(bbox.xmin + side - 1, width - 1.f));
		bbox.ymax = round(min(bbox.ymin + side - 1, height - 1.f));
	}
}


/******************************************************
// ������:GenerateBBox
// ˵��: ����Bbox
// ����:�ŷ�
// ʱ��:2017.11.4
// ��ע:
/*******************************************************/
void MTCNN_ONET::GenerateBBox(Blob<float>* confidence, Blob<float>* reg_box,
	float scale, float thresh) {
	int feature_map_w_ = confidence->width();
	int feature_map_h_ = confidence->height();
	int spatical_size = feature_map_w_*feature_map_h_;
	const float* confidence_data = confidence->cpu_data() + spatical_size; // �������������
	const float* reg_data = reg_box->cpu_data();
	candidate_boxes_.clear();
	for (int i = 0; i < spatical_size; i++) {
		if (confidence_data[i] >= thresh) {
			// feature_map��Ӧ����λ��
			int y = i / feature_map_w_;
			int x = i - feature_map_w_ * y;
			FaceInfos FaceInfos;
			// FaceBox &faceBox = FaceInfos.bbox;

			//ԭͼscale�ˣ�����scale����
			FaceInfos.bbox.xmin = (float)(x * pnet_stride) / scale;
			FaceInfos.bbox.ymin = (float)(y * pnet_stride) / scale;
			FaceInfos.bbox.xmax = (float)(x * pnet_stride + pnet_cell_size - 1.f) / scale;
			FaceInfos.bbox.ymax = (float)(y * pnet_stride + pnet_cell_size - 1.f) / scale;

			FaceInfos.bbox_reg[0] = reg_data[i];
			FaceInfos.bbox_reg[1] = reg_data[i + spatical_size];
			FaceInfos.bbox_reg[2] = reg_data[i + 2 * spatical_size];
			FaceInfos.bbox_reg[3] = reg_data[i + 3 * spatical_size];

			FaceInfos.bbox.score = confidence_data[i];
			candidate_boxes_.push_back(FaceInfos);
		}
	}
}


/******************************************************
// ������:MTCNN_ONET
// ˵��: ���캯��
// ����:�ŷ�
// ʱ��:����ģ��
// ��ע:
/*******************************************************/
MTCNN_ONET::MTCNN_ONET() {
	Caffe::set_mode(Caffe::GPU);
	PNet_.reset(new Net<float>(("./models/det1.prototxt"), TEST));
	PNet_->CopyTrainedLayersFrom("./models/det1.caffemodel");
	RNet_.reset(new Net<float>(("./models/det2.prototxt"), TEST));
	RNet_->CopyTrainedLayersFrom("./models/det2.caffemodel");
	ONet_.reset(new Net<float>(("./models/det3-half.prototxt"), TEST));
	ONet_->CopyTrainedLayersFrom("./models/det3-half.caffemodel");

	// �������Ϊ3ͨ��ͼƬ
	Blob<float>* input_layer = PNet_->input_blobs()[0];
	int num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3) << "Input layer should have 3 channels.";
}



/******************************************************
// ������:ProposalNet
// ˵��:PNet
// ����:�ŷ�
// ʱ��:2017.11.4
// ��ע:Pnet���̣����ɳ�����ѡ
/*******************************************************/
vector<FaceInfos> MTCNN_ONET::ProposalNet(const cv::Mat& img, int minSize, float threshold, float factor) {
	cv::Mat  resized;
	int width = img.cols;
	int height = img.rows;
	float scale = 12.f / minSize;
	float minWH = std::min(height, width) *scale;
	std::vector<float> scales;
	while (minWH >= 12) { // �߶Ƚ�����
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}
	Blob<float>* input_layer = PNet_->input_blobs()[0];
	total_boxes_.clear(); //��face��Ŀ
	// ͼ��߶Ƚ����������resize��forward
	for (int i = 0; i < scales.size(); i++) { 
		int ws = (int)std::ceil(width*scales[i]);
		int hs = (int)std::ceil(height*scales[i]);
		cv::resize(img, resized, cv::Size(ws, hs), 0, 0, cv::INTER_LINEAR);
		input_layer->Reshape(1, 3, hs, ws);
		PNet_->Reshape();
		float * input_data = input_layer->mutable_cpu_data();
		cv::Vec3b * img_data = (cv::Vec3b *)resized.data;
		int spatial_size = ws* hs;
		for (int k = 0; k < spatial_size; ++k) {  // opencv----->blobs
			input_data[k] = float((img_data[k][0] - mean_val)* std_val);
			input_data[k + spatial_size] = float((img_data[k][1] - mean_val) * std_val);
			input_data[k + 2 * spatial_size] = float((img_data[k][2] - mean_val) * std_val);
		}
		PNet_->Forward();

		Blob<float>* confidence = PNet_->blob_by_name("prob1").get();
		Blob<float>* reg = PNet_->blob_by_name("conv4-2").get();
		GenerateBBox(confidence, reg, scales[i], threshold); // �õ���ѡ����
		std::vector<FaceInfos> bboxes_nms = NMS(candidate_boxes_, 0.5, 'u'); //�Ǽ���ֵ����
		if (bboxes_nms.size() > 0) {
			total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
		}
	}
	int num_box = (int)total_boxes_.size();
	vector<FaceInfos> res_boxes;
	//���г߶ȵĶ���ɺ󣬽���nms
	if (num_box != 0) {
		res_boxes = NMS(total_boxes_, 0.7f, 'u');
		BBoxRegression(res_boxes);
		BBoxPadSquare(res_boxes, width, height);
	}
	return res_boxes;
}



/******************************************************
// ������:NextStage
// ˵��:Rnet
// ����:�ŷ�
// ʱ��:2017.11.4
// ��ע:����Pnet,������һ���׶����ɵ�proposal
/*******************************************************/
vector<FaceInfos> MTCNN_ONET::NextStage(const cv::Mat& image, vector<FaceInfos> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold) {
	vector<FaceInfos> res;
	int batch_size = (int)pre_stage_res.size();
	if (batch_size == 0)
		return res;
	Blob<float>* input_layer = nullptr;
	Blob<float>* confidence = nullptr;
	Blob<float>* reg_box = nullptr;
	Blob<float>* reg_landmark = nullptr;

	switch (stage_num) {
	case 2: {
		input_layer = RNet_->input_blobs()[0];
		input_layer->Reshape(batch_size, 3, input_h, input_w); //N*3*24*24
		RNet_->Reshape();
	}break;
	case 3: {
		input_layer = ONet_->input_blobs()[0];
		input_layer->Reshape(batch_size, 3, input_h, input_w);
		ONet_->Reshape();
	}break;
	default:
		return res;
		break;
	}
	float * input_data = input_layer->mutable_cpu_data();
	int spatial_size = input_h*input_w;

#pragma omp parallel for num_threads(threads_num)
	for (int n = 0; n < batch_size; ++n) {
		FaceBox &box = pre_stage_res[n].bbox;
		Mat roi = image(Rect(Point((int)box.xmin, (int)box.ymin), Point((int)box.xmax, (int)box.ymax))).clone();
		resize(roi, roi, Size(input_w, input_h));
		float *input_data_n = input_data + input_layer->offset(n);
		Vec3b *roi_data = (Vec3b *)roi.data;
		CHECK_EQ(roi.isContinuous(), true);
		for (int k = 0; k < spatial_size; ++k) {
			input_data_n[k] = float((roi_data[k][0] - mean_val)*std_val);
			input_data_n[k + spatial_size] = float((roi_data[k][1] - mean_val)*std_val);
			input_data_n[k + 2 * spatial_size] = float((roi_data[k][2] - mean_val)*std_val);
		}
	}
	switch (stage_num) {
	case 2: {
		RNet_->Forward();
		confidence = RNet_->blob_by_name("prob1").get();
		reg_box = RNet_->blob_by_name("conv5-2").get();
	}break;
	case 3: {
		ONet_->Forward();
		confidence = ONet_->blob_by_name("prob1").get();
		reg_box = ONet_->blob_by_name("conv6-2").get();
		reg_landmark = ONet_->blob_by_name("conv6-3").get();
	}break;
	}
	const float* confidence_data = confidence->cpu_data();
	const float* reg_data = reg_box->cpu_data();
	const float* landmark_data = nullptr;
	if (reg_landmark) {
		landmark_data = reg_landmark->cpu_data();
	}
	for (int k = 0; k < batch_size; ++k) {
		if (confidence_data[2 * k + 1] >= threshold) {
			FaceInfos info;
			info.bbox.score = confidence_data[2 * k + 1];
			info.bbox.xmin = pre_stage_res[k].bbox.xmin;
			info.bbox.ymin = pre_stage_res[k].bbox.ymin;
			info.bbox.xmax = pre_stage_res[k].bbox.xmax;
			info.bbox.ymax = pre_stage_res[k].bbox.ymax;
			for (int i = 0; i < 4; ++i) {
				info.bbox_reg[i] = reg_data[4 * k + i];
			}
			if (reg_landmark) {
				float w = info.bbox.xmax - info.bbox.xmin + 1.f;
				float h = info.bbox.ymax - info.bbox.ymin + 1.f;
				for (int i = 0; i < 5; ++i){
					info.landmark[2 * i] = landmark_data[10 * k + 2 * i] * w + info.bbox.xmin;
					info.landmark[2 * i + 1] = landmark_data[10 * k + 2 * i + 1] * h + info.bbox.ymin;
				}
			}
			res.push_back(info);
		}
	}
	return res;
}


/******************************************************
// ������:Detect
// ˵��:mtcnn���������������������ں���
// ����:�ŷ�
// ʱ��:2017.11.04
// ��ע:
/*******************************************************/
vector<FaceInfos> MTCNN_ONET::Detect(const cv::Mat& image, const int minSize, const float* threshold, const float factor, const int stage) {
	vector<FaceInfos> pnet_res;
	vector<FaceInfos> rnet_res;
	vector<FaceInfos> onet_res;

	if (stage >= 1){
		// pNet
		pnet_res = ProposalNet(image, minSize, threshold[0], factor);
	}

	//Rnet
	if (stage >= 2 && pnet_res.size()>0){
		if (pnet_max_detect_num < (int)pnet_res.size()){
			pnet_res.resize(pnet_max_detect_num);
		}
		int num = (int)pnet_res.size();
		int size = (int)ceil(1.f*num / step_size);
		for (int iter = 0; iter < size; ++iter){
			int start = iter*step_size;
			int end = min(start + step_size, num);
			// ÿ�δ����һ�����mini_batch��rect
			vector<FaceInfos> input(pnet_res.begin() + start, pnet_res.begin() + end); 
			vector<FaceInfos> res = NextStage(image, input, 24, 24, 2, threshold[1]);//Rnet
			rnet_res.insert(rnet_res.end(), res.begin(), res.end());
		}
		rnet_res = NMS(rnet_res, 0.7f, 'u');
		BBoxRegression(rnet_res);
		BBoxPadSquare(rnet_res, image.cols, image.rows);
	}

	//Onet �е�ع�
	if (stage >= 3 && rnet_res.size()>0){
		int num = (int)rnet_res.size();
		int size = (int)ceil(1.f*num / step_size);
		for (int iter = 0; iter < size; ++iter){
			int start = iter*step_size;
			int end = min(start + step_size, num);
			vector<FaceInfos> input(rnet_res.begin() + start, rnet_res.begin() + end);
			vector<FaceInfos> res = NextStage(image, input, 48, 48, 3, threshold[2]);
			onet_res.insert(onet_res.end(), res.begin(), res.end());
		}
		BBoxRegression(onet_res);
		onet_res = NMS(onet_res, 0.7f, 'm');
		BBoxPad(onet_res, image.cols, image.rows);

	}
	if (stage == 1){
		return pnet_res;
	}
	else if (stage == 2){
		return rnet_res;
	}
	else if (stage == 3){
		return onet_res;
	}
	else{
		return onet_res;
	}
}
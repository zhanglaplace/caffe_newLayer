#ifndef  _FACE_DETECT_H__
#define  _FACE_DETECT_H__


#include <caffe/caffe.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <boost/shared_ptr.hpp>

using namespace caffe;
using namespace std;
using namespace cv;

//omp
const int threads_num = 4;
//pnet config
const float pnet_stride = 2;
const float pnet_cell_size = 12;
const int pnet_max_detect_num = 5000;
//mean & std
const float mean_val = 127.5f;
const float std_val = 0.0078125f;
//minibatch size
const int step_size = 128;

typedef struct FaceBox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
} FaceBox;
typedef struct FaceInfos {
	float bbox_reg[4];
	float landmark_reg[10];
	float landmark[10];
	FaceBox bbox;
} FaceInfos;

class MTCNN_ONET {
public:
	MTCNN_ONET();
	vector<FaceInfos> Detect(const cv::Mat& img, const int min_size, const float* threshold, const float factor, const int stage);
protected:
	vector<FaceInfos> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
	vector<FaceInfos> NextStage(const cv::Mat& image, vector<FaceInfos> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
	void BBoxRegression(vector<FaceInfos>& bboxes);
	void BBoxPadSquare(vector<FaceInfos>& bboxes, int width, int height);
	void BBoxPad(vector<FaceInfos>& bboxes, int width, int height);
	void GenerateBBox(Blob<float>* confidence, Blob<float>* reg_box, float scale, float thresh);
	std::vector<FaceInfos> NMS(std::vector<FaceInfos>& bboxes, float thresh, char methodType);
	float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);

private:
	boost::shared_ptr<Net<float>> PNet_;
	boost::shared_ptr<Net<float>> RNet_;
	boost::shared_ptr<Net<float>> ONet_;

	std::vector<FaceInfos> candidate_boxes_;
	std::vector<FaceInfos> total_boxes_;
};

#endif
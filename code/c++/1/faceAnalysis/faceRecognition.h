#ifndef _FACE_RECOGNITION_H_
#define _FACE_RECOGNITION_H_


#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace caffe;

class FaceRecognition
{
public:
	FaceRecognition();
	float getSimilarity(const cv::Mat& lhs, const cv::Mat& rhs, bool useFligImg = true);
	float getSimilarity(const std::vector<float>& lhs, const std::vector<float>& rhs);
	float getSimilarity(cv::Mat& lImg, cv::Mat& rImg, int k, bool useFligImg = true);
	float getMoldLength(const std::vector<float>& vec);
	vector<float>getLastLayerFeatures(const cv::Mat& _img); //������������
	vector<float>getLastLayerFeaturesFlip(const cv::Mat& _img);//ƴ���������
	void getLastLayerFeaturesFlip(const cv::Mat& _img, int k);//ƴ���������
	float getSimilarity(std::vector<cv::Mat>& _vecs, bool useFligImg = true);
	int getMaxRecordNum();//��ȡ������ڵ�ע������
	void saveFeaturesToFile(int id, char* name);
	void loadTotalFacesFromFile(char* fileName, vector<vector<float> >&);
	void UpdateTotalFeature();
	vector<vector<float> >registeredFaces;
	int compareTwoVectors();
	void getNameFromId(char* name, int id);
private:
	boost::shared_ptr<Net<float> > _net;
	vector<float>feature; // ��ǰע���˵�����
	const int feature_len;
};



#endif // !_FACE_RECOGNITION_H_

#ifndef _FACE_RECOGNITION_H_
#define _FACE_RECOGNITION_H_


#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
using namespace caffe;

class FaceRecognition
{
public:
	FaceRecognition();
	// ��ȡcnn���������������ƶ�
	int compareTwoVectors(); // Ĭ�ϱȽϵ�ǰ�����Ϳ�����
	float getL2length(const vector<float>& vec);
	void getLastLayerFeaturesFlip(const cv::Mat& _img, int k);
	vector<float> getLastLayerFeaturesFlip(const cv::Mat& _img);
	float getSimilarity(const std::vector<float>& lhs, const std::vector<float>& rhs);
	////��ȡ������ڵ�ע������
	int  getMaxRecordNum(); // ��ǰ���е�ע������
	void saveFeaturesToFile(int id, char* name); // �����������ļ�
	void loadTotalFacesNameFromFile(char* fileName, std::map<string, int>&);
	void loadTotalFacesFeatureFromFile(char* fileName, vector<vector<float> >&);// ��������
	void updateTotalFeatureName(std::string sName); // ��������������
	const char* getNameFromId(int id); //����id��ȡ��ǰ��������
	int  checkName(std::string name);
	

private:
	vector<float>feature_; // ��ǰע���˵�����

public:
	const int feature_len_; // �������� 
	float threshold_;// ����ʶ����ֵ
	boost::shared_ptr<Net<float> > _net;
	vector<vector<float> >totalFeature_; // ������
	map<string, int> totalName_;
	int total_num_;
};






#endif // !_FACE_RECOGNITION_H_

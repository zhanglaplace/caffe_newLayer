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
	// 获取cnn特征并求特征相似度
	int compareTwoVectors(); // 默认比较当前特征和库特征
	float getL2length(const vector<float>& vec);
	void getLastLayerFeaturesFlip(const cv::Mat& _img, int k);
	vector<float> getLastLayerFeaturesFlip(const cv::Mat& _img);
	float getSimilarity(const std::vector<float>& lhs, const std::vector<float>& rhs);
	////获取最大现在的注册人数
	int  getMaxRecordNum(); // 当前已有的注册人数
	void saveFeaturesToFile(int id, char* name); // 保存特征到文件
	void loadTotalFacesNameFromFile(char* fileName, std::map<string, int>&);
	void loadTotalFacesFeatureFromFile(char* fileName, vector<vector<float> >&);// 加载特征
	void updateTotalFeatureName(std::string sName); // 更新特征库特征
	const char* getNameFromId(int id); //根据id获取当前人物姓名
	int  checkName(std::string name);
	

private:
	vector<float>feature_; // 当前注册人的特征

public:
	const int feature_len_; // 特征长度 
	float threshold_;// 人脸识别阈值
	boost::shared_ptr<Net<float> > _net;
	vector<vector<float> >totalFeature_; // 库特征
	map<string, int> totalName_;
	int total_num_;
};






#endif // !_FACE_RECOGNITION_H_

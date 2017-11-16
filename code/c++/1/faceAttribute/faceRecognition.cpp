#include "stdafx.h"
#include "faceRecognition.h"



void trim(std::string &s)
{
	if (s.empty())
	{
		return ;
	}
	s.erase(0, s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ") + 1);
}

/******************************************************
// FaceRecognition
// ���캯��,����model ����������
// �ŷ�
// 2017.07.26
/*******************************************************/
FaceRecognition::FaceRecognition() :feature_len_(1024),threshold_(0.25){
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif // CPU_ONLY
	_net.reset(new Net<float>("./models/sphereface_deploy.prototxt", caffe::TEST));
	_net->CopyTrainedLayersFrom("./models/sphereface_model.caffemodel");

	loadTotalFacesFeatureFromFile("./db/vsFaceDb.txt", totalFeature_);
	loadTotalFacesNameFromFile("./db/fileNameID.txt", totalName_);
}

/******************************************************
// ������:compareTwoVectors
// ˵��: ��ǰע���������������������б����Ƚ�
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
int FaceRecognition::compareTwoVectors(){

	int totalSize = totalFeature_.size();
	float similarity = -1.f;
	int maxSimilarityId = -1;
	float maxSimilarity = -1.f;
	for (int i = 0; i < totalSize; i++){
		similarity = getSimilarity(std::vector<float>(feature_.begin() + 1, feature_.end()), std::vector<float>(totalFeature_[i].begin() + 1, totalFeature_[i].end()));
		if (maxSimilarity < similarity){
			maxSimilarity = similarity;
			maxSimilarityId = i;
		}
	}
	if (maxSimilarity < threshold_){ //�������ƶȶ�С����ֵ
		maxSimilarityId = -1;
	}
	return maxSimilarityId;
}

/******************************************************
// ������:getSimilarity
// ˵��:�Ƚ��������������ƶ�
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
float FaceRecognition::getSimilarity(const vector<float>& lhs, const std::vector<float>& rhs){
	assert(lhs.size(), rhs.size());
	assert(lhs.size(), feature_len_);
	float tmp = 0.0;
	for (int i = 0; i < feature_len_; i++){
		tmp += lhs[i] * rhs[i];
	}

	return tmp / (getL2length(lhs) * getL2length(rhs));
}

/******************************************************
// ������:getL2length
// ˵��:��ȡһ��������ģ��
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
float FaceRecognition::getL2length(const vector<float>& vec){
	assert(vec.size());
	float tmp = 0.f;
	for (int i = 0; i != vec.size(); ++i){
		tmp += vec[i] * vec[i];
	}
	return sqrt(tmp);
}

/******************************************************
// ������:getLastLayerFeaturesFlip
// ˵��: ��ȡͼ���CNN����
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void FaceRecognition::getLastLayerFeaturesFlip(const cv::Mat& _img,int k){

	cv::Mat img = _img.clone();
	img.convertTo(img, CV_32FC3);
	resize(img, img, cvSize(96, 112));
	Blob<float>* inputBlob = _net->input_blobs()[0];
	int width = inputBlob->width();
	int height = inputBlob->height();
	int spatial_dim = width*height;
	img = (img - 127.5)*0.0078125; //��ȥ��ֵ���ٹ�һ
	float* data = inputBlob->mutable_cpu_data();
	assert(img.channels() == 3);
	cv::Vec3f* img_data = (cv::Vec3f*)img.data;
	for (int k = 0; k < spatial_dim; ++k){
		data[k] = (float)img_data[k][0];
		data[k + spatial_dim] = (float)img_data[k][1];
		data[k + 2 * spatial_dim] = (float)img_data[k][2];
	}
	Blob<float>* outputBlob = _net->Forward()[0];
	const float* begin = outputBlob->cpu_data();
	const float* end = begin + outputBlob->num()*outputBlob->channels();
	feature_ = vector<float>(begin, end);
	feature_.insert(feature_.begin(), k);
}


/******************************************************
// ������:loadTotalFacesFeatureFromFile
// ˵��:���ļ����������˵���������
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void FaceRecognition::loadTotalFacesFeatureFromFile(char* fileName, vector<vector<float> >& totalFeature){

	std::ifstream featureFile(fileName, std::ios_base::in);
	if (!featureFile){
		return;
	}
	vector<float> feat(feature_len_ + 1, 0);
	string line;
	totalFeature.swap(vector<vector<float> >()); //�û�Ϊ��
	while (!featureFile.eof()){ //����
		getline(featureFile, line);
		stringstream ss(line);
		for (int i = 0; i < feature_len_ + 1; i++){
			ss >> feat[i];
		}
		totalFeature.push_back(feat);
	}
	featureFile.close();
}


/******************************************************
// ������:getNameFromId
// ˵��:����id���ļ���ȡ������
// ����:�ŷ�
// ʱ��:2015.11.15
// ��ע:
/*******************************************************/
const char* FaceRecognition::getNameFromId(int id){

	map<string, int>::iterator it = totalName_.begin();
	while (it != totalName_.end()){
		if (it->second == id){
			return it->first.c_str();
		}
	}
	return NULL;


	/* ���������ı�����
	FILE* idFile = 0;
	idFile = fopen("./db/fileNameID.txt", "r");

	if (!idFile)
		return;
	int readId = -1;
	int fileResult = -1;
	char readName[50];

	do
	{
		fileResult = fscanf(idFile, "%d %s\n", &readId, &readName);
		if (id == readId)
		{
			sprintf(name, "%s", readName);
			fclose(idFile);
			return;
		}

	} while (fileResult != EOF);
	*/
}

/******************************************************
// ������:UpdateTotalFeature
// ˵��: �������е�������������
// ����:�ŷ�
// ʱ��:2015.11.15
// ��ע:
/*******************************************************/
void FaceRecognition::updateTotalFeatureName(string sName){

	totalFeature_.push_back(feature_); // ����������
	total_num_ = total_num_ + 1; 
	totalName_.insert(make_pair(sName, total_num_));
}


/******************************************************
// ������:getMaxRecordNum
// ˵��:�鿴Ŀǰע�����������
// ����:�ŷ�
// ʱ��:2017.11.14
// ��ע:
/*******************************************************/
int FaceRecognition::getMaxRecordNum(){
	/*
	int maxNum = 0;
	FILE* fin = fopen("./db/fileNameID.txt", "r");
	if (!fin) // ���ļ�
		return maxNum;
	int returnVal = 0, num = 0;
	char isim[50];
	do
	{
		returnVal = fscanf(fin, "%d", &num);
		fscanf(fin, "%s", isim);
		if (num > maxNum)
			maxNum = num;
	} while (returnVal != EOF);
	fclose(fin);
	*/
	return total_num_;
}


/******************************************************
// ������:saveFeatureToFile
// ˵��:���������������ļ���
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void FaceRecognition::saveFeaturesToFile(int id, char* isim){

	std::ofstream featureFile("./db/vsFaceDb.txt", std::ios_base::app);
	std::ofstream fileNameID("./db/fileNameID.txt", std::ios_base::app);
	for (int i = 0; i < feature_.size(); i++){
		featureFile << feature_[i] << " ";
	}
	featureFile << "\n";
	featureFile.close();
	fileNameID << id << " " << isim << std::endl;
	featureFile.close();
	fileNameID.close();
}


/******************************************************
// ������:loadTotalFacesNameFromFile
// ˵��:���ļ��м����������ݿ������
// ����:�ŷ�
// ʱ��:2017.11.14
// ��ע:
/*******************************************************/
void FaceRecognition::loadTotalFacesNameFromFile(char* fileName, std::map<std::string, int>& sName){

	// ���
	sName.swap(map<string, int>());

	std::ifstream fin(fileName,std::ios_base::in);
	std::string lines;
	while (getline(fin,lines)){
		string::size_type position = lines.find(" ");
		if (position != lines.npos){
			sName.insert(std::make_pair(lines.substr(position + 1),atoi(lines.substr(0, position).c_str())));
		}
		else{
			return;
		}
	}
	fin.close();
	total_num_ = sName.size();
}



/******************************************************
// ������:checkName
// ˵��:�ж�����������Ƿ����
// ����:�ŷ�
// ʱ��:
// ��ע:�Ѿ����ڷ��� 1 ��Ϊ�շ���-1 ����ȷ����0
/*******************************************************/
int FaceRecognition::checkName(string sName){
	string tmp = sName;
	trim(tmp);
	if (tmp.empty()){ //�����ϳ�����ַ���
		return -1;
	}
	else if (totalName_.find(tmp) != totalName_.end()){ // �Ѿ����������ݿ�
		return 1;
	}
	else{
		return 0; // ��Ч��
	}
}
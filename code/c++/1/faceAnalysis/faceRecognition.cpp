#include "stdafx.h"
#include "FaceRecognition.h"

/******************************************************
// FaceRecognition
// 构造函数,加载model
// 张峰
// 2017.07.26
/*******************************************************/
FaceRecognition::FaceRecognition():feature_len(1024){
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif // CPU_ONLY
	_net.reset(new Net<float>("./models/sphereface_deploy.prototxt", caffe::TEST));
	_net->CopyTrainedLayersFrom("./models/sphereface_model.caffemodel");

	loadTotalFacesFromFile("vsFaceDb.txt", registeredFaces);
}


/******************************************************
// 函数名:getMaxRecordNum
// 说明:查看目前注册的任务数量
// 作者:张峰
// 时间:2017.11.14
// 备注:
/*******************************************************/
int FaceRecognition::getMaxRecordNum(){
	int maxNum = 0;
	FILE* fin = fopen("fileNameID.txt", "r");
	if (!fin) // 无文件
		return maxNum;
	int returnVal = 0,num = 0;
	char isim[50];
	do
	{
		returnVal = fscanf(fin, "%d", &num);
		fscanf(fin, "%s", isim);
		if (num > maxNum) 
			maxNum = num;
	} while (returnVal != EOF);
	fclose(fin);
	return maxNum;
}


/******************************************************
// 函数名:saveFeatureToFile
// 说明:保存特征向量到文件中
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void FaceRecognition::saveFeaturesToFile(int id, char* isim){

	std::ofstream featureFile("vsFaceDb.txt", std::ios_base::app);
	std::ofstream fileNameID("fileNameID.txt", std::ios_base::app);
	for (int i = 0; i < feature.size(); i++){
		featureFile << feature[i] << " ";
	}
	featureFile << "\n";
	featureFile.close();
	fileNameID << id << " " << isim << std::endl;
	featureFile.close();
	fileNameID.close();
	/*
	//追加形式
	FILE* featureFile = fopen("vsFaceDb.txt", "a");
	FILE* fileNameID = fopen("fileNameID.txt", "a");

	fprintf(featureFile, "%d ", id);
	for (int  i = 0; i < feature.size(); i++){
		fprintf(featureFile, "%f ", feature[i]);
	}
	fclose(featureFile);
	fprintf(fileNameID, "%d %s\n", id, isim); //标号和姓名
	fclose(fileNameID);
	*/

}

/******************************************************
// getSimilarity
// 计算特征向量的相似度(余弦距离)
// 张峰
// 2017.07.26
/*******************************************************/
float FaceRecognition::getSimilarity(const std::vector<float>& lhs, const std::vector<float>& rhs){
	int num = lhs.size();
	assert(num == rhs.size());
	float tmp = 0.0;
	for (int i = 0; i != num; ++i){
		tmp += lhs[i] * rhs[i];
	}
	return tmp / (getMoldLength(lhs)*getMoldLength(rhs));
}

/******************************************************
// getSimilarity
// 计算两幅图像特征的相似度
// 张峰
// 2017.07.26
/*******************************************************/
float FaceRecognition::getSimilarity(const cv::Mat& lhs, const cv::Mat& rhs, bool useFligImg /* = true */){
	vector<float>feat1, feat2;
	if (useFligImg){//使用镜像
		feat1 = getLastLayerFeaturesFlip(lhs);
		feat2 = getLastLayerFeaturesFlip(rhs);
	}
	else
	{
		feat1 = getLastLayerFeatures(lhs);
		feat2 = getLastLayerFeatures(rhs);
	}
	//return std::max<float>(0, getSimilarity(feat1, feat2));
	return getSimilarity(feat1, feat2);//需要考虑正负
}



/******************************************************
// getMoldLength
// 计算向量的模长
// 张峰
// 2017.07.26
/*******************************************************/
float FaceRecognition::getMoldLength(const std::vector<float>& vec){
	int num = vec.size();
	float tmp = 0.0;
	for (int i = 0; i != num; ++i){
		tmp += vec[i] * vec[i];
	}
	return sqrt(tmp);
}


/******************************************************
// getLastLayerFeatures
// 获取图像的特征
// 张峰
// 2017.07.26
/*******************************************************/
vector<float> FaceRecognition::getLastLayerFeatures(const cv::Mat& _img){
	cv::Mat img = _img.clone();
	img.convertTo(img, CV_32FC3);

	Blob<float>* inputBlob = _net->input_blobs()[0];
	int width = inputBlob->width();
	int height = inputBlob->height();
	int index = 0;
	resize(img, img, cv::Size(width, height));
	img = (img - 127.5)*0.0078125; //减去均值后再归一
	float* data = inputBlob->mutable_cpu_data();
	assert(img.channels() == 3);
	for (int k = 0; k < img.channels(); ++k){
		for (int i = 0; i < height; ++i){
			for (int j = 0; j < width; ++j){
				index = (k*height + i)*width + j;
				data[index] = img.at<cv::Vec3f>(i, j)[k];
			}
		}
	}
	//vector<Blob<float>*> inputs(1, inputBlob); 
	//clock_t t1 = clock();
	const vector<Blob<float>*> outputBlobs = _net->Forward();
	Blob<float>* outputBlob = outputBlobs[0];
	const float* begin = outputBlob->cpu_data();
	const float* end = begin + outputBlob->channels();

	/*
	cv::Mat imgR;
	cv::flip(img, imgR, 1);
	float* dataR = inputBlob->mutable_cpu_data();
	for (int k = 0; k < imgR.channels(); ++k){
	for (int i = 0; i < height; ++i){
	for (int j = 0; j < width; ++j){
	index = (k*height + i)*width + j;
	dataR[index] = imgR.at<cv::Vec3f>(i, j)[k];
	}
	}
	}
	const vector<Blob<float>*> outputBlobsR = _net->Forward();
	Blob<float>* outputBlobR = outputBlobsR[0];
	const float* beginR = outputBlobR->cpu_data();
	const float* endR = begin + outputBlobR->channels();
	vector<float>featureR = vector<float>(beginR, endR);
	for (int i = 0; i < featureR.size(); ++i){
	featureL.push_back(featureR[i]);
	}
	*/
	//std::cout << clock() - t1 << "ms\n";
	return vector<float>(begin, end);
}





/******************************************************
// getLastLayerFeaturesFlip
// 获取图像的拼接特征
// 张峰
// 2017.07.26
//center loss中，是将原图的特征向量和左右翻转图的特征向量拼接起来，返回最终的特征向量
/*******************************************************/
vector<float> FaceRecognition::getLastLayerFeaturesFlip(const cv::Mat& _img){
	vector<float>res1 = getLastLayerFeatures(_img);
	cv::Mat flipImg;
	cv::flip(_img, flipImg, 1);//centerLoss采用图像翻转上下连接,sphereface采用上下翻转，但是做了归一化
	vector<float>res2 = getLastLayerFeatures(flipImg);
	assert(res1.size(), res2.size());
	for (int i = 0; i < res2.size(); ++i){
		res1.push_back(res2[i]);
	}
	return res1;
}



/******************************************************
// getSimilarity
// 获取两幅图像的相似度
// 张峰
// 2017.07.26
// 获取图像的相似度，以组为单位输入网络
/*******************************************************/
float FaceRecognition::getSimilarity(std::vector<cv::Mat>& _vecs, bool useFligImg /* = true */){

	Blob<float>* inputBlob = _net->input_blobs()[0];
	int width = inputBlob->width();
	int height = inputBlob->height();
	int nchannels = inputBlob->channels();
	int nums = inputBlob->num();
	int n = _vecs.size();
	for (int i = 0; i < n; ++i){
		_vecs[i].convertTo(_vecs[i], CV_32FC3);
		resize(_vecs[i], _vecs[i], cv::Size(width, height));
		_vecs[i] = (_vecs[i] - 127.5)*0.0078125; //减去均值后再归一
	}
	vector<cv::Mat>flip;
	vector<cv::Mat>im;
	if (useFligImg){
		for (int i = 0; i < n; ++i){
			cv::Mat flipImg;
			cv::flip(_vecs[i], flipImg, 0);
			im.push_back(_vecs[i]);
			im.push_back(flipImg);
		}
	}
	else
	{
		im = _vecs;
	}
	assert(im.size(), nums);
	int index = 0;
	float* data = inputBlob->mutable_cpu_data();
	//assert(im.channels() == 3);

	for (int n = 0; n < nums; ++n){
		for (int k = 0; k < nchannels; ++k){
			for (int i = 0; i < height; ++i){
				for (int j = 0; j < width; ++j){
					index = ((n* nchannels + k)*height + i)*width + j;
					data[index] = im[n].at<cv::Vec3f>(i, j)[k];
				}
			}
		}
	}
	clock_t t1 = clock();
	Blob<float>* outputBlob = _net->Forward()[0];
	const float* begin = outputBlob->cpu_data();
	const float* end = begin + nums / 2 * outputBlob->channels();
	vector<float>feat1(begin, end);
	vector<float>feat2(end, end + nums / 2 * outputBlob->channels());
	//std::cout << clock() - t1 << "ms\n";
	//return std::max<float>(0, getSimilarity(feat1, feat2));
	return getSimilarity(feat1, feat2);
	//return std::vector<float>(begin, end);
}



/******************************************************
//getSimilarity
// 获取两幅图像的相似度,以fliplayer模式
// 张峰
// 2017.07.26
// 获取图像的相似度，以组为单位输入网络
/*******************************************************/
float FaceRecognition::getSimilarity(cv::Mat& lImg, cv::Mat& rImg, int k, bool useFligImg /* = true */){
	vector<float>feat1, feat2;
	feat1 = getLastLayerFeaturesFlip(lImg);
	feat2 = getLastLayerFeaturesFlip(rImg);
	/*
	std::cout << feat1.size() << feat2.size() << std::endl;

	for (int i = 0; i < feat1.size();++i){
	std::cout << feat1[i] << std::endl;
	}
	*/

	return getSimilarity(feat1, feat2);
}


/******************************************************
// getLastLayerFeaturesFlip
// 获取图像的拼接特征
// 张峰
// 2017.07.26
// center loss中，是将原图的特征向量和左右翻转图的特征向量拼接起来，返回最终的特征向量
// 此处使用的是添加fliplayer层次的方式
/*******************************************************/
void FaceRecognition::getLastLayerFeaturesFlip(const cv::Mat& _img, int k){
	cv::Mat img = _img.clone();
	img.convertTo(img, CV_32FC3);

	Blob<float>* inputBlob = _net->input_blobs()[0];
	int width = inputBlob->width();
	int height = inputBlob->height();
	//int index = 0;
	//resize(img, img, cv::Size(width, height)); 
	img = (img - 127.5)*0.0078125; //减去均值后再归一
	float* data = inputBlob->mutable_cpu_data();
	assert(img.channels() == 3);
	
	//for (int k = 0; k < img.channels(); ++k){
	//	for (int i = 0; i < height; ++i){
	//		for (int j = 0; j < width; ++j){
	//			index = (k*height + i)*width + j;
	//			data[index] = img.at<cv::Vec3f>(i, j)[k];
	//		}
	//	}
	//}
	int spatial_dim = width*height;
	cv::Vec3f* img_data = (cv::Vec3f*)img.data;
	for (int k = 0; k < spatial_dim; ++k){
		data[k] = (float)img_data[k][0];
		data[k + spatial_dim] = (float)img_data[k][1];
		data[k + 2 * spatial_dim] = (float)img_data[k][2];
	}

	//vector<Blob<float>*> inputs(1, inputBlob); 
	//clock_t t1 = clock();
	//const vector<Blob<float>*> outputBlobs = _net->Forward();
	//输出的为2*512*1*1
	Blob<float>* outputBlob = _net->Forward()[0];
	const float* begin = outputBlob->cpu_data();
	const float* end = begin + outputBlob->num()*outputBlob->channels();
	feature =  vector<float>(begin, end);
	feature.insert(feature.begin(), k);
}


/******************************************************
// 函数名:loadTotalFacesFromFile
// 说明:从文件加在现有人的人脸特征
// 作者:张峰
// 时间:
// 备注:
/*******************************************************/
void FaceRecognition::loadTotalFacesFromFile(char* fileName, vector<vector<float> >& totalFeature){

	std::ifstream featureFile(fileName, std::ios_base::in);
	if (!featureFile){
		return;
	}
	vector<float> feat(feature_len+1,0);
	string line;
	totalFeature.swap(vector<vector<float> >()); //置换为空
	while (!featureFile.eof()){ //遍历
		getline(featureFile, line);
		stringstream ss(line);
		for (int i = 0; i < feature_len + 1; i++){
			ss >> feat[i];
		}
		totalFeature.push_back(feat);
	}
	featureFile.close();
}



/******************************************************
// 函数名:compareTwoVectors
// 说明:当前人脸与
// 作者:张峰
// 时间:
// 备注:
/*******************************************************/
int FaceRecognition::compareTwoVectors(){
	
	int totalSize = registeredFaces.size();
	float similarity = -1.f;
	// 总人
	int maxSimilarityId = 0;
	float maxSimilarity = -1.f;
	for (int i = 0; i < totalSize; i++){
		similarity = getSimilarity(std::vector<float>(feature.begin() + 1, feature.end()), std::vector<float>(registeredFaces[i].begin() + 1,registeredFaces[i].end()));
		if (maxSimilarity < similarity){
			maxSimilarity = similarity;
			maxSimilarityId = i;
		}
	}
	return maxSimilarityId;

}

/******************************************************
// 函数名:getNameFromId
// 说明:根据id从文件中取出姓名
// 作者:张峰
// 时间:2015.11.15
// 备注:
/*******************************************************/
void FaceRecognition::getNameFromId(char* name, int id){

	FILE* idFile = 0;
	idFile = fopen("fileNameID.txt", "r");

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
}

/******************************************************
// 函数名:UpdateTotalFeature
// 说明: 更新已有的人物特征向量
// 作者:张峰
// 时间:2015.11.15
// 备注:
/*******************************************************/
void FaceRecognition::UpdateTotalFeature(){
	
	registeredFaces.push_back(feature); 
}
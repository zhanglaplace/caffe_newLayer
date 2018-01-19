#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <io.h>
#include <direct.h>
using namespace std;

#define TYPECLASS 7
#define Target_Image_Width  144
#define Target_Image_Height 144
#define LANDMARK_SIZE_DOUBLE 68

vector<string>facial_expression{ "Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger" };
vector<cv::Point2d>target_points{ cv::Point2d(45.4419, 66.4667), cv::Point2d(98.2977, 66.2161), cv::Point2d(72.0378, 92.2327), cv::Point2d(50.3240, 118.7556), cv::Point2d(94.0919, 118.5481) };

using namespace std;
using namespace cv;

const std::string num2Emotion[7] = { "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise" };

using std::string;

/* Pair (label, confidence) representing a prediction. */


cv::Point3f transform(cv::Point3f pt, cv::Mat rot, cv::Point3f trans) {
	cv::Point3f res;
	res.x = rot.at<float>(0, 0)*pt.x + rot.at<float>(0, 1)*pt.y + rot.at<float>(0, 2)*pt.z + trans.x;
	res.y = rot.at<float>(1, 0)*pt.x + rot.at<float>(1, 1)*pt.y + rot.at<float>(1, 2)*pt.z + trans.y;
	res.z = rot.at<float>(2, 0)*pt.x + rot.at<float>(2, 1)*pt.y + rot.at<float>(2, 2)*pt.z + trans.z;
	return res;
}

/******************************************************
// 函数名:findNonReflectiveTransform
// 说明:happynear对齐方式
// 作者:张峰
// 时间:
// 备注:
/*******************************************************/
cv::Mat findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, Mat& Tinv = Mat()) {
	assert(source_points.size() == target_points.size());
	assert(source_points.size() >= 2);
	Mat U = Mat::zeros(target_points.size() * 2, 1, CV_64F);
	Mat X = Mat::zeros(source_points.size() * 2, 4, CV_64F);
	for (int i = 0; i < target_points.size(); i++) {
		U.at<double>(i * 2, 0) = source_points[i].x;
		U.at<double>(i * 2 + 1, 0) = source_points[i].y;
		X.at<double>(i * 2, 0) = target_points[i].x;
		X.at<double>(i * 2, 1) = target_points[i].y;
		X.at<double>(i * 2, 2) = 1;
		X.at<double>(i * 2, 3) = 0;
		X.at<double>(i * 2 + 1, 0) = target_points[i].y;
		X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
		X.at<double>(i * 2 + 1, 2) = 0;
		X.at<double>(i * 2 + 1, 3) = 1;
	}
	Mat r = X.inv(DECOMP_SVD)*U;
	Tinv = (Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
		r.at<double>(1), r.at<double>(0), 0,
		r.at<double>(2), r.at<double>(3), 1);
	Mat T = Tinv.inv(DECOMP_SVD);
	Tinv = Tinv(Rect(0, 0, 2, 3)).t();
	return T(Rect(0, 0, 2, 3)).t();
}


/******************************************************
// 函数名:findSimilarityTransform
// 说明:happynear对齐方式五点
// 作者:张峰
// 时间:
// 备注:
/*******************************************************/
cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, Mat& Tinv = Mat()) {
	Mat Tinv1, Tinv2;
	Mat trans1 = findNonReflectiveTransform(source_points, target_points, Tinv1);
	std::vector<Point2d> source_point_reflect;
	for (auto sp : source_points) {
		source_point_reflect.push_back(Point2d(-sp.x, sp.y));
	}
	Mat trans2 = findNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
	trans2.colRange(0, 1) *= -1;
	std::vector<Point2d> trans_points1, trans_points2;
	transform(source_points, trans_points1, trans1);
	transform(source_points, trans_points2, trans2);
	double norm1 = norm(Mat(trans_points1), Mat(target_points), NORM_L2);
	double norm2 = norm(Mat(trans_points2), Mat(target_points), NORM_L2);
	Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
	return norm1 < norm2 ? trans1 : trans2;
}

std::vector<cv::Point2d> getVertexFromBox(cv::Rect box) {
	return{ Point2d(box.x, box.y), Point2d(box.x + box.width, box.y), Point2d(box.x + box.width, box.y + box.height), Point2d(box.x, box.y + box.height) };
}

/******************************************************
// SplitString
// 分割字符串
// 张峰
// 2017.07.26
/*******************************************************/
void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}


/******************************************************
// getFaceScores
// 从文件中读取图像对的类别与scores
// 张峰
// 2017.07.26
/*******************************************************/
void getFaceScores(std::vector<float>& scores, std::vector<int>& cls, const string& scoresAndClass){
	scores.swap(vector<float>());
	cls.swap(vector<int>());
	ifstream fin(scoresAndClass, ios_base::in);
	string lines;
	while (getline(fin, lines)){
		vector<string>line;
		SplitString(lines, line, "\t");
		scores.push_back(atof(line[0].c_str()));
		cls.push_back(atoi(line[1].c_str()));
	}
	fin.close();
	cout << scores.size() << endl;
}




/******************************************************
// getAccuracy
// 计算准确率
// 张峰
// 2017.07.26
//
/*******************************************************/
float getAccuracy(int thrNum, const string& scoresAndClass){
	vector<float> scores;
	vector<int> cls;
	getFaceScores(scores, cls, scoresAndClass);
	vector<float>accuracy(2 * thrNum + 1, 0.0);
	int pairNum = scores.size();
	float maxac = 0, maxacth = 0;
	for (int i = 0; i < 2 * thrNum + 1; ++i){
		float tmp = -1.0 + i*1.0 / thrNum;
		int correctNum = 0;
		for (int j = 0; j < pairNum; ++j){
			if (((scores[j] > tmp) && (cls[j] == 1)) || ((scores[j] < tmp) && (cls[j] == 0)))
				correctNum++;
		}
		accuracy[i] = static_cast<float>(correctNum) / pairNum;
		if (maxac < accuracy[i]){
			maxac = accuracy[i];
			maxacth = tmp;
		}
	}
	std::cout << "best ac is :" << maxac << " the threshold is :" << maxacth << std::endl;
	return maxac;
}

/******************************************************
// affine_warp
// 图像的五点仿射变换-happynear
// 张峰
// 2017.07.26
//
/*******************************************************/
void affine_warp(){
	cv::Point2d pt1(105.8306, 109.8005);
	cv::Point2d pt2(147.9323, 112.5533);
	cv::Point2d pt3(121.3533, 139.1172);
	cv::Point2d pt4(106.1169, 155.6359);
	cv::Point2d pt5(144.3622, 156.3451);
	vector<cv::Point2d>ts;
	ts.push_back(pt1);   ts.push_back(pt2);   ts.push_back(pt3);    ts.push_back(pt4);  ts.push_back(pt5);

	cv::Point2d pS1(30.2946, 51.6963);
	cv::Point2d pS2(65.5318, 51.5014);
	cv::Point2d pS3(48.0252, 71.7366);
	cv::Point2d pS4(33.5493, 92.3655);
	cv::Point2d pS5(62.7299, 92.2041);
	vector<cv::Point2d>target_points;
	target_points.push_back(pS1);   target_points.push_back(pS2);   target_points.push_back(pS3);    target_points.push_back(pS4);  target_points.push_back(pS5);



	Mat trans_inv;
	Mat trans = findSimilarityTransform(ts, target_points, trans_inv);
	Mat cropImage;

	Mat img = imread("D:/software/caffe/caffe-master/FaceRec/mtcnn/models/Jennifer_Aniston_0016.jpg");

	warpAffine(img, cropImage, trans, Size(96, 112));
	//imshow("cropImage", cropImage);
	vector<Point2d> rotatedVertex;
	transform(getVertexFromBox(Rect(0, 0, 96, 112)), rotatedVertex, trans_inv);
	for (int i = 0; i < 4; i++)
		line(img, rotatedVertex[i], rotatedVertex[(i + 1) % 4], Scalar(0, 255, 0), 2);
	imshow("t", img);
	waitKey(0);

}




/******************************************************
// phase_affect_datasets
// 解析affectNet并为训练做准备
// 张峰
// 2017.09.26
//
/*******************************************************/
void phase_affect_datasets(){
	const string srcName = "F:/datasets/face_expression/AffectNet/affectNet_validation.csv"; //原文件
	const string dstName = "F:/datasets/face_expression/AffectNet/affectNet_validation.txt"; //保存的txt img,class_type face_x face_y face_x+face_w face_y+face_h
	const string tmpImg = "F:/datasets/face_expression/AffectNet/Manually_Annotated_Images-5pts-Train/"; // 保存生成后的目录
	const string dirPath = "F:/datasets/face_expression/AffectNet/Manually_Annotated_Images/"; //原始图片路径
	ifstream fin(srcName.c_str(), ios_base::in);
	ofstream fou(dstName.c_str(), ios_base::out);
	if (!fin) {
		cout << "open file  " << srcName << "  error\n";
		return ;
	}

	Mat img, cropImage,trans;
	string lines, imgName, target_filename, tmp_folder;
	int count = 0; // 统计样本个数，每隔1000个样本打印一次
	int class_type[7] = { 0 };
	Point2d left_eye, right_eye, tip_nose, left_mouth, right_mouth;
	getline(fin, lines);//第一行不需要
	while (getline(fin, lines)){
		vector<string>line;
		
		SplitString(lines, line, ",");
		if (line.size() != 9){
			cout << "phase " << lines << "error\n";
			return ;
		}
		// 只考虑7种基本的表情
		if (atoi(line[6].c_str()) >= 7) {
			continue;
		}

		// 读取图片，保存
		imgName = dirPath + line[0];
		img = imread(imgName.c_str());
		target_filename = tmpImg + line[0];
		tmp_folder = target_filename.substr(0, target_filename.find_last_of('/'));
		if (_access(tmp_folder.c_str(), 0) == -1){
			_mkdir(tmp_folder.c_str());
		}



		//生成对应的点，并进行对象的仿射变换
		vector<string>landmark_pts;
		SplitString(line[5], landmark_pts, ";");
		if (landmark_pts.size() != 136 ){
			continue;
		}
		//切分得到对应的点位置


		for (int i = 72; i < 83;){//左眼
			left_eye.x += atof(landmark_pts[i].c_str());
			left_eye.y += atof(landmark_pts[i+1].c_str());
			i += 2;
		}

		for (int i = 84; i < 95;){//右眼
			right_eye.x += atof(landmark_pts[i].c_str());
			right_eye.y += atof(landmark_pts[i + 1].c_str());
			i += 2;
		}
		left_eye /= 6;
		right_eye /= 6;
		
		// 鼻尖
		tip_nose.x = atof(landmark_pts[60].c_str());
		tip_nose.y = atof(landmark_pts[61].c_str());

		// 左嘴角
		left_mouth.x = atof(landmark_pts[96].c_str());
		left_mouth.y = atof(landmark_pts[97].c_str());

		// 右嘴角
		right_mouth.x = atof(landmark_pts[108].c_str());
		right_mouth.y = atof(landmark_pts[109].c_str());
		
		vector<Point2d> facial_points{ left_eye, right_eye, tip_nose, left_mouth, right_mouth };

		trans = findSimilarityTransform(facial_points, target_points);
		warpAffine(img, cropImage, trans, Size(144,144));
		imwrite(target_filename, cropImage);

		left_mouth.x = 0, left_mouth.y = 0, left_eye.x = 0, left_eye.y = 0, tip_nose.x = 0, tip_nose.y = 0, right_mouth.x = 0, right_mouth.y = 0, right_eye.x = 0, right_eye.y = 0;

		// 写入txt并统计各个类别的数据
		fou << target_filename << " " << facial_expression[atoi(line[6].c_str())]  << endl;
		class_type[atoi(line[6].c_str())]++;

		// 有效的表情数量,1000输出一次
		count++;
		if (count % 100 == 0) {
			cout << "finishing " << count << " images..........\n";
		}
	}
	fou.close();
	fin.close();

	cout << "total images:" << count << endl;
	for (size_t i = 0; i < TYPECLASS; i++) {
		cout << facial_expression[i] << ": " << class_type[i] << endl;
	}
}


void loadTotalFacesNameFromFile(char* fileName, std::map<std::string, int>& sName){

	// 清空
	sName.swap(map<string, int>());

	std::ifstream fin(fileName, std::ios_base::in);
	std::string lines;
	while (getline(fin, lines)){
		string::size_type position = lines.find(" ");
		if (position != lines.npos){
			sName.insert(std::make_pair(lines.substr(position + 1), atoi(lines.substr(0, position).c_str())));
		}
		else{
			return;
		}
	}
	fin.close();
}

/******************************************************
// 函数名:warpface
// 说明:68点对齐
// 作者:张峰
// 时间:
// 备注:
/*******************************************************/
void warpface(){



}



//test opencv functions 
int main(){

	phase_affect_datasets();
	system("pause");
	return 0;

}
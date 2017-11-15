#ifndef _UTILS_H__
#define _UTILS_H__
#include <iostream>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <fstream>

typedef std::map<float, int> MAP;


typedef struct tdata{
	cv::Mat oImg;
	cv::Mat invImg;
	int M;
	int N;
}tdata;


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
// PlotROCOpencv
// 用Opencv画出ROC曲线
// 张峰
// 2017.07.26
// 
/*******************************************************/
void PlotROCOpencv(){
	std::ifstream fin("./results/rocPts_cos.txt", std::ios_base::in);
	std::vector<cv::Point>pts;
	std::string lines;
	while (std::getline(fin, lines)){
		std::vector<std::string>line;
		SplitString(lines, line, "\t");
		pts.push_back(cv::Point(512 * atof(line[0].c_str()), 512 - 512 * atof(line[1].c_str())));
	}
	cv::Mat img(513, 513, CV_32FC1);
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; ++j){
			img.at<uchar>(i, j) = 0;
		}
	}
	cv::circle(img, pts[0], 1, CV_RGB(255, 255, 255));
	for (int i = 1; i < pts.size(); ++i)
	{
		cv::line(img, pts[i - 1], pts[i], CV_RGB(0, 0, 255), 1);
	}
	cv::imshow("img", img);
	imwrite("./results/img_roc_Flip_cos.jpg", img);
	cv::waitKey(0);
	fin.close();
}


/******************************************************
// myfunOne
// 类别是否为1
// 张峰
// 2017.07.26
/*******************************************************/
bool myfunOne(const std::pair<float, int>& vec){
	return vec.second == 1;
}

//排序适配器
bool compareScore(const std::pair<float, int> &lhs, const std::pair<float, int> &rhs){
	return lhs.first > rhs.first;
}



/******************************************************
// getFaceScores
// 从文件中读取图像对的类别与scores
// 张峰
// 2017.07.26
/*******************************************************/
void getFaceScores(std::vector<float>& scores, std::vector<int>& cls){
	scores.swap(vector<float>());
	cls.swap(vector<int>());
	ifstream fin("./results/LfwScoreAndLabelWithFlip_cos.txt", ios_base::in);
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
// plotROC
// 绘制roc曲线
// 张峰
// 2017.07.26
// 计算ROC曲线中的FPR和TPR，并绘制图像
/*******************************************************/
void plotROC(){
	std::vector<float> scores;
	std::vector<int> cls;
	getFaceScores(scores, cls);
	std::ofstream fou("./results/rocPts_cos.txt", std::ios_base::out);
	//绘制ROC曲线
	std::vector <std::pair<float, int>>tVector;
	std::vector<cv::Point>pts;
	for (int curr = 0; curr != scores.size(); ++curr){
		tVector.push_back(std::make_pair(scores[curr],cls[curr]));
	}
	sort(tVector.begin(), tVector.end(), compareScore);//分数从高到低排序                  
	for (auto curr = tVector.begin(); curr != tVector.end(); ++curr){
		int TP = std::count_if(tVector.begin(), curr+1, myfunOne);
		int FP = curr+1 - tVector.begin() - TP;
		int FN = std::count_if(curr+1, tVector.end(), myfunOne);
		int TN = tVector.end() - curr-1 - FN;
		float FPR = static_cast<float>(FP) / (FP + TN);
		float TPR = static_cast<float>(TP) / (TP + FN);
		fou << FPR << "\t" << TPR << std::endl;
		pts.push_back(cv::Point(FPR, TPR));
	}
	fou.close();
	PlotROCOpencv();

}


/******************************************************
// getAccuracy
// 计算准确率
// 张峰
// 2017.07.26
//
/*******************************************************/
vector<float>getAccuracy(int thrNum){
	vector<float> scores;
	vector<int> cls;
	getFaceScores(scores, cls);
	vector<float>accuracy(2 * thrNum + 1, 0.0);
	int pairNum = scores.size();
	for (int i = 0; i < 2 * thrNum + 1; ++i){
		float tmp = -1.0 + i*1.0 / thrNum;
		int correctNum = 0;
		for (int j = 0; j < pairNum; ++j){
			if (((scores[j] > tmp) && (cls[j] == 1)) || ((scores[j] < tmp) && (cls[j] == 0)))
				correctNum++;
		}
		accuracy[i] = static_cast<float>(correctNum) / pairNum;
	}
	return accuracy;
}

/******************************************************
// getShift
// Point2f2中心点的偏移
// 张峰
// 2017.07.26
//
/*******************************************************/
cv::Point2f getShift(vector<cv::Point2f>& pts){
	int tol = 1000;
	cv::Point2f minPoint2f(INT_MAX, INT_MAX);
	cv::Point2f maxPoint2f(0.0, 0.0);
	cv::Point2f centerPoint2f(0.0, 0.0);
	cv::Point2f spanPoint2f(0.0, 0.0);
	int n = pts.size();
	for (int i = 0; i < n; ++i){//求出x和y的最小值和最大值
		if (pts[i].x < minPoint2f.x)
			minPoint2f.x = pts[i].x;
		if (pts[i].y < minPoint2f.y)
			minPoint2f.y = pts[i].y;
		if (pts[i].x > maxPoint2f.x)
			maxPoint2f.x = pts[i].x;
		if (pts[i].y > maxPoint2f.y)
			maxPoint2f.y = pts[i].y;
	}
	centerPoint2f.x = (minPoint2f.x + maxPoint2f.x) / 2;
	centerPoint2f.y = (minPoint2f.y + maxPoint2f.y) / 2;
	spanPoint2f.x = maxPoint2f.x - minPoint2f.x;
	spanPoint2f.y = maxPoint2f.y - minPoint2f.y;
	if ((spanPoint2f.x > 0 && abs(centerPoint2f.x) / spanPoint2f.x > tol) || (spanPoint2f.y > 0 && abs(centerPoint2f.y) / spanPoint2f.y > tol))
		return centerPoint2f;
	return cv::Point2f(0.0, 0.0);
}

/******************************************************
// maketform
// 仿射变换结构体
// 张峰
// 2017.07.26
//
/*******************************************************/
tdata maketform(cv::Mat& tranM, const string method = "affine"){
	cv::Mat _tranM = tranM.clone();
	int height = _tranM.rows;
	int width = _tranM.cols;
	int N = width - 1;
	tdata tmp;//保存原图A和A-1，最后一列置001
	tmp.oImg = tranM.clone();
	tmp.invImg = tmp.oImg.inv();
	for (int i = 0; i < height;++i){
		tmp.invImg.at<float>(i, width - 1) = 0.0;
	}
	tmp.invImg.at<float>(height - 1, width - 1) = 1.0;

	tmp.M = N;
	tmp.N = N;
	return tmp;
}



/******************************************************
// print
// 打印MAT方便观察
// 张峰
// 2017.07.26
//
/*******************************************************/
void print(cv::Mat& img){
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			cout << img.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;
}


/******************************************************
// findNonreflectiveSimilarity
// 五点相似转换--非反射版本
// 张峰
// 2017.07.26
//
/*******************************************************/
tdata findNonreflectiveSimilarity(const vector<cv::Point2f>&fac, const vector<cv::Point2f>& ori){
	int M = fac.size();
	cv::Mat matrix = Mat::ones(2 * M, 4, CV_32FC1);
	for (int i = 0; i < M; ++i){ //A 10*4
		matrix.at<float>(i, 0) = ori[i].x;
		matrix.at<float>(i, 1) = ori[i].y;
		matrix.at<float>(i, 2) = 1.0;
		matrix.at<float>(i, 3) = 0.0;
		matrix.at<float>(i + M, 0) = ori[i].y;
		matrix.at<float>(i + M, 1) = -ori[i].x;
		matrix.at<float>(i + M, 2) = 0.0;
		matrix.at<float>(i + M, 3) = 1.0;
	}
	
	cv::Mat _vecfac(2 * M, 1, CV_32FC1);
	for (int i = 0; i < M; ++i){//B  10*1
		_vecfac.at<float>(i, 0) = fac[i].x;
		_vecfac.at<float>(i + M, 0) = fac[i].y;
	}
	//计算AX=B
	cv::Mat transMatrix(4, 1, CV_32FC1);
	cv::solve(matrix, _vecfac, transMatrix, DECOMP_NORMAL); // solve (Ax=b) for x 
	cv::Mat tinv(3, 3, CV_32FC1);
	float sc = transMatrix.at<float>(0, 0);
	float ss = transMatrix.at<float>(1, 0);
	float tx = transMatrix.at<float>(2, 0);
	float ty = transMatrix.at<float>(3, 0);
	//scale and trans
	tinv.at<float>(0, 0) = sc;
	tinv.at<float>(1, 0) = ss;
	tinv.at<float>(2, 0) = tx;
	tinv.at<float>(0, 1) = -ss;
	tinv.at<float>(1, 1) = sc;
	tinv.at<float>(2, 1) = ty;
	tinv.at<float>(0, 2) = 0;
	tinv.at<float>(1, 2) = 0;
	tinv.at<float>(2, 2) = 1;
	tinv = tinv.inv();//矩阵求逆
	tinv.at<float>(0, 2) = 0;
	tinv.at<float>(1, 2) = 0;
	tinv.at<float>(2, 2) = 1;
	return maketform(tinv);
}



/******************************************************
// trans_affine
// 根据放射矩阵对五点进行放射变换
// 张峰
// 2017.07.26
//
/*******************************************************/
cv::Mat trans_affine(cv::Mat& img, tdata& t, const string& method){
	Mat M;
	if (method == "forward")
		M = t.oImg.clone();//5*3
	else
		M = t.invImg.clone();
	//cout << img.rows << "\t" << img.cols << endl;
	Mat x1 = cv::Mat::ones(img.rows, img.cols + 1, CV_32FC1);
	for (int i = 0; i < x1.rows;++i){//img的右部做扩充
		for (int j = 0; j < x1.cols - 1;++j){
			x1.at<float>(i, j) = img.at<float>(i, j);//其余地方设置为1.0
		}
	}
	Mat U1 = x1*M;//5*3  和 3*3;
	return cv::Mat(U1,cv::Rect(0,0,U1.cols-1,U1.rows));
}


/******************************************************
// tformfwd
// 转换接口
// 张峰
// 2017.07.26
//
/*******************************************************/
cv::Mat tformfwd(tdata& trans, const vector<cv::Point2f>& fac){
	tdata t = trans;
	cv::Mat tmp(fac.size(), 2, CV_32FC1);
	for (int i = 0; i < fac.size(); ++i){ //A 10*4
		tmp.at<float>(i, 0) = fac[i].x;
		tmp.at<float>(i, 1) = fac[i].y;
	}
	int inputDim = t.M;//输入空间维度
	int outputDim = t.N;//输出空间维度
	int D = fac.size();//长度为5
	int L = 2;
	Mat U = tmp.clone();//这个是输入的原始坐标
	return trans_affine(U, trans, "forward");

}


/******************************************************
// findSimilarity
// 五点相似转换
// 张峰
// 2017.07.26
//
/*******************************************************/
tdata findSimilarity(vector<cv::Point2f>&fac, vector<cv::Point2f>& ori){
	int M = fac.size();
	tdata trans1 = findNonreflectiveSimilarity(fac, ori);

	vector<cv::Point2f>oriR(ori.size(), cv::Point2f(0.0,0.0));
	for (int i = 0; i < oriR.size();++i){//所有的x乘以-1；
		oriR[i].x = -1*ori[i].x;
		oriR[i].y = ori[i].y;
	}
	tdata trans2r = findNonreflectiveSimilarity(fac, oriR);

	cv::Mat treflectY(3, 3, CV_32FC1);
	for (int i = 0; i < 3;++i){
		treflectY.at<float>(i, 0) = -1 * trans2r.oImg.at<float>(i, 0);
		treflectY.at<float>(i, 1) = trans2r.oImg.at<float>(i, 1);
		treflectY.at<float>(i, 2) = trans2r.oImg.at<float>(i, 2);
	}
	tdata trans2 = maketform(treflectY);

	//对齐中心五点
	cv::Mat xy(ori.size(), 2, CV_32FC1);
	for (int i = 0; i < ori.size(); ++i){ //A 10*4
		xy.at<float>(i, 0) = ori[i].x;
		xy.at<float>(i, 1) = ori[i].y;
	}
	// Figure out if trans1 or trans2 is better
	//转换后的五点
	//对齐中心五点
	Mat xy1 = tformfwd(trans1,fac);//这个地方的计算是存在问题的 ,xy和fac都没问题， 计算的trans1是存在问题的
	float norm1 = cv::norm(cv::Mat(xy1 - xy));
	Mat xy2 = tformfwd(trans2, fac);
	float norm2 = norm(xy2 - xy);
	if (norm1 <= norm2)
		return trans1;
	return trans2;
}



/******************************************************
//  imtransform
// 根据求解转换矩阵,对图像进行转化和resize
// 张峰
// 2017.07.26
//
/*******************************************************/
Mat imtransform(cv::Mat& img, tdata& td, cv::Size& cropSize){
	string interpolant = "linear";
	string padmethod = "fill";
	const int stanardFrequency = 1000, first_prop_arg = 3;
	float halfwidth = 1.0;
	int n = floor(halfwidth*stanardFrequency);
	vector<float>x_kernel(n + 1, 0.0);
	float lr = halfwidth / n;
	for (int i = 0; i <= n;++i){
		x_kernel[i] = 1 - i*lr;
	}
	//xdata
	int x_data = cropSize.width;//1*96
	int y_data = cropSize.height;//1*112
	cv::Size imgSize(img.cols,img.rows);//96*112
	int u_data = imgSize.width;
	int v_data = imgSize.height;
	int tdims_A = 2;
	int tdims_B = 2;
	cv::Size tsize_b(img.rows, img.cols);//112*96
	Mat gridImg(y_data,x_data, CV_32FC2);//112*96*2 grid
	for (int i = 0; i < y_data; i++){
		for (int j = 0; j < x_data; ++j){
			gridImg.at<cv::Vec2f>(i, j)[0] = j+1;
			gridImg.at<cv::Vec2f>(i, j)[1] = i+1;
		}
	}
	Mat reshapeUimg = gridImg.reshape(1, x_data*y_data);
	Mat xImg = trans_affine(reshapeUimg, td, "inv_transform");
	Mat X(y_data, x_data, CV_32FC2);//112*96*2 grid

	for (int i = 0; i < y_data; i++){ 
		for (int j = 0; j < x_data; ++j){ // x转置会和matlab一样
			int index = i*x_data + j;
			X.at<cv::Vec2f>(i, j)[0] = xImg.at<float>(index, 0);
			X.at<cv::Vec2f>(i, j)[1] = xImg.at<float>(index,1);
		}
	}
	//此处差一个maltab的    function B = resample( A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R ) 具体的实现resamp_fcn

	return cv::Mat();
}


/******************************************************
// cp2tform
// 五点相似转换求出转换矩阵,并对图像进行相似转换。
// 张峰
// 2017.07.26
// 接口为原始五点与目标五点，代码依照matlab修改。
/*******************************************************/
Mat cp2tform(vector<cv::Point2f>& facial5Point2fs, vector<cv::Point2f>& coord5Point2fs, cv::Mat& img, cv::Size cropSize){
	assert(facial5Point2fs.size() == coord5Point2fs.size());
	cv::Point2f facialshift = getShift(facial5Point2fs);
	cv::Point2f coordshift = getShift(coord5Point2fs);
	bool needToshift = (facialshift.x != 0 || facialshift.y != 0 || coordshift.x != 0 || coordshift.y != 0);
	tdata tfm;
	if (!needToshift){//如果不需要shift
		tfm = findSimilarity(facial5Point2fs, coord5Point2fs);
	}
	else{//需要shift
		for (int i = 0; i < coord5Point2fs.size();++i){
			facial5Point2fs[i].x = facial5Point2fs[i].x - facialshift.x;
			facial5Point2fs[i].y = facial5Point2fs[i].y - facialshift.y;
			coord5Point2fs[i].x = coord5Point2fs[i].x - coordshift.x;
			coord5Point2fs[i].y = coord5Point2fs[i].y - coordshift.y;
		}
		tfm = findSimilarity(facial5Point2fs, coord5Point2fs);
	}
	//imtransform 可以用函数warpAffine来代替，需要将tfm转化为Mat
	Mat transfm = tfm.oImg.t()(cv::Range(0,2),cv::Range(0,3));
	Mat dstImg;
	cv::warpAffine(img, dstImg, transfm, cropSize);
	return dstImg;
}


#endif

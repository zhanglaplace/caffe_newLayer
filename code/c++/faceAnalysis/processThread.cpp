#include "stdafx.h"
#include "processThread.h"

using namespace std;
using namespace cv;




inline DWORD GETTICKCOUNTDIFFERENCE(DWORD first, DWORD second)
{
	return (((first) <= (second)) ? ((second)-(first)) : (UINT_MAX - (first)+(second)));
}


/******************************************************
// 函数名: ProcessThread
// 说明: 人脸检测线程构造函数
// 作者:张峰
// 时间:2017.11.14 
// 备注: 
/*******************************************************/
ProcessThread::ProcessThread(int nThreadPriority, void* dlgPtr, Mat& faceImg)
: Thread(nThreadPriority)
, m_bReadyForNewImage(false)
, m_nCounter(0)
, m_dvLastUpdate(0){
	InitializeCriticalSection(&m_CriticalSection);
	InitializeCriticalSection(&m_CriticalSectionForFaceImg);
	m_dlgPtr = dlgPtr;

#ifndef CPU_ONLY
	Caffe::set_mode(Caffe::GPU);
#else
	Caffe::set_mode(Caffe::CPU);
#endif
	mtcnn = new MTCNN_ONET();
	capture = VideoCapture(0);
	threshold[0] = 0.7;
	threshold[1] = 0.6;
	threshold[2] = 0.6;
}


/******************************************************
// 函数名:findFaceAndDrawRect
// 说明:人脸检测与显示
// 作者:张峰
// 时间:2017.11.14
// 备注:
/*******************************************************/
void ProcessThread::findFaceAndDrawRect(Mat& img, bool& willRecord){
	Mat tmp = img.clone();
	resize(tmp, tmp, cvSize(tmp.cols/ 2, tmp.rows / 2));
	vector<FaceInfos> faces = mtcnn->Detect(tmp, 40, threshold, 0.709, 3);
	if (faces.size() > 0){
		for (int i = 0; i < faces.size(); i++){
			int x = 2*(int)faces[i].bbox.xmin;
			int y = 2*(int)faces[i].bbox.ymin;
			int w = 2 * (int)(faces[i].bbox.xmax - faces[i].bbox.xmin + 1);
			int h = 2 * (int)(faces[i].bbox.ymax - faces[i].bbox.ymin + 1);
			cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
		}
		for (int i = 0; i < faces.size(); i++){
			float *landmark = faces[i].landmark;
			for (int j = 0; j < 5; j++){
				cv::circle(img, cv::Point(2*(int)landmark[2 * j], 2*(int)landmark[2 * j + 1]), 1, cv::Scalar(255, 255, 0), 2);
			}
		}
	}


}


/******************************************************
// 函数名:ProcessJob
// 说明:人脸检测
// 作者:张峰
// 时间:2017.11.14
// 备注:
/*******************************************************/
void ProcessThread::ProcessJob(){
	Mat frame;
	capture >> frame;
	findFaceAndDrawRect(frame, willRecord);
}

/******************************************************
// 函数名:Go
// 说明:现成
// 作者:张峰
// 时间:
// 备注:
/*******************************************************/
UINT ProcessThread::Go(){

	DWORD	dwTickPrevious = GetTickCount();
	DWORD	dwDiff = 0;
	DWORD	dwCurrent = 0;
	DWORD	dwTimeToWait = (DWORD)(1000.0 / FRAMES_TO_PROCESS_IN_A_SEC);

	while (!GetExitRequest())
	{
		dwCurrent = GetTickCount();
		dwDiff = GETTICKCOUNTDIFFERENCE(dwTickPrevious, dwCurrent);

		if (dwDiff < dwTimeToWait)
		{
			WakableSleep(dwTimeToWait - dwDiff);
			continue;
		}
		else dwTickPrevious = dwCurrent;

		ProcessJob();
	}
	return 0;
}
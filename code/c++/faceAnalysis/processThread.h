#ifndef _PROCESS_THREAD_H_
#define _PROCESS_THREAD_H_

#define IMAGE_HANDLED_MESSAGE  (WM_APP + 1)
#define FRAMES_TO_PROCESS_IN_A_SEC (20)

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "thread.h"
#include <queue>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include <omp.h>
#include "faceDetect.h"


class ProcessThread:public Thread
{
public:
	ProcessThread(int nThreadPriority = THREAD_PRIORITY_ABOVE_NORMAL, void* dlgPtr = 0,cv::Mat& faceImg = cv::Mat());
	//~ProcessThread();
	//bool isValidImage(Mat&);
	void findFaceAndDrawRect(Mat& img, bool& willRecord);
	void ProcessJob();
public:
	bool willRecord;
	bool wasSuccesfullyRecorded;
	CRITICAL_SECTION	m_CriticalSectionForFaceImg;
private:
	VideoCapture capture;
	Mat faceImg;
	//Mat getImageFromCam();
	double scale;
	void* m_dlgPtr;
	MTCNN_ONET* mtcnn;
	float threshold[3];

protected:

	virtual UINT Go();
	volatile bool		m_bReadyForNewImage;
	volatile int		m_nCounter;
	CRITICAL_SECTION	m_CriticalSection;

	DWORD				m_dvLastUpdate;
	bool				m_running;
};

#endif
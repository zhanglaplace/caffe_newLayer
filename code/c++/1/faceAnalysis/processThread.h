#ifndef _PROCESS_THREAD_H_
#define _PROCESS_THREAD_H_

#define IMAGE_HANDLED_MESSAGE  (WM_APP + 1)


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

#define FRAMES_TO_PROCESS_IN_A_SEC (20)

class ProcessThread:public Thread
{
public:

	ProcessThread(int nThreadPriority = THREAD_PRIORITY_ABOVE_NORMAL, void* dlgPtr = 0, IplImage *faceIm = 0);
	~ProcessThread();
	void ProcessJob();
	bool isValidImage(Mat& );
public:
	bool willRecord;
	bool wasSuccesfullyRecorded;
	CRITICAL_SECTION	m_CriticalSectionForFaceImg;
private:
	VideoCapture capture;
	IplImage* dlgfaceImage;
	//Mat getImageFromCam();
	double scale;
	void* m_dlgPtr;
	MTCNN_ONET* mtcnn;
	float threshold[3];
	void findFaceAndDrawRect(Mat& img, bool& willRecord);

protected:

	virtual UINT Go();
	CRITICAL_SECTION	m_CriticalSection;
};

#endif
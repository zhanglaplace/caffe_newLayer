
// faceAnalysisDlg.h : 头文件
//

#pragma once
#include "register.h"
#include "processThread.h"
#include "RegisterBox.h"
#include "faceRecognition.h"
#include "atltypes.h"
#include "D:\Deeplearning\Caffe\caffe\3rdparty\Release\include\opencv2\highgui\highgui.hpp"

// CfaceAnalysisDlg 对话框
class CfaceAnalysisDlg : public CDialogEx
{

private:
	ProcessThread* faceDetectThread;
	FaceRecognition* frObj;

// 构造
public:
	CfaceAnalysisDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_FACEANALYSIS_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持

// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnCamera();
	afx_msg void OnBnClickedCapture();
	afx_msg void OnBnClickedClosecamera();
	afx_msg void OnBnClickedBtnRegesiter();
	afx_msg void OnBnClickedBtnRecognition();
	afx_msg void OnBnClickedCancel();
	void showImage(Mat& im,UINT ID);
	CString m_name;
	HDC m_picHdc;
	CRect m_picRect;
	// 摄像头是否打开
	bool m_isStart;
	// 是否拍照注册
	bool m_isRecord;
	// 是否识别
	bool m_isRecognition;
	// 关闭摄像头
	bool m_isClose;
public:
	// 摄像头
	VideoCapture m_capture;
	HDC m_capHdc;
	CRect m_capRect;
	IplImage* faceImage;
	Mat faceImgShow;
	Mat captureImg;//register图片
	Mat frame;//摄像头图片
};

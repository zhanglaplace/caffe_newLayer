
// faceAttributeDlg.h : 头文件
//

#pragma once
#include "register.h"
#include "atltypes.h"
#include <opencv2/opencv.hpp>
#include "faceDetect.h"
#include "faceRecognition.h"
#include "RegisterBox.h"
// CfaceAttributeDlg 对话框
class CfaceAttributeDlg : public CDialogEx
{
// 构造
public:
	CfaceAttributeDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_FACEATTRIBUTE_DIALOG };

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
	// 显示图像窗口DC
	HDC m_picHdc;
	// 显示摄像头图片区域rect
	CRect m_picRect;
	// 注册DC
	HDC m_recordHdc;
	CRect m_recordRect;

	// 摄像头是否开启
	bool m_isStart;
	// 是否拍照
	bool m_isRecord;

	// 是否识别
	bool m_isRecognition;

	// 摄像头
	VideoCapture m_capture;
	// 注册图片
	Mat m_captureImg;

	//人脸检测指针
	FaceDetect* m_fd;
	FaceRecognition* m_fr;

	void showImage(Mat& im, UINT ID);
	afx_msg void OnBnClickedBtnClosecameta();
	afx_msg void OnBnClickedBtnOpencameta();
	afx_msg void OnBnClickedBtnCapture();
	bool m_isLandmark;
	afx_msg void OnBnClickedRegister();
	afx_msg void OnClose();
	afx_msg void OnBnClickedRecognition();
	CString m_resultName;
};

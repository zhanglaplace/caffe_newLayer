
// faceAnalysisDlg.h : ͷ�ļ�
//

#pragma once
#include "register.h"
#include "processThread.h"
#include "RegisterBox.h"
#include "faceRecognition.h"
#include "atltypes.h"
#include "D:\Deeplearning\Caffe\caffe\3rdparty\Release\include\opencv2\highgui\highgui.hpp"

// CfaceAnalysisDlg �Ի���
class CfaceAnalysisDlg : public CDialogEx
{

private:
	ProcessThread* faceDetectThread;
	FaceRecognition* frObj;

// ����
public:
	CfaceAnalysisDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_FACEANALYSIS_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��

// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
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
	// ����ͷ�Ƿ��
	bool m_isStart;
	// �Ƿ�����ע��
	bool m_isRecord;
	// �Ƿ�ʶ��
	bool m_isRecognition;
	// �ر�����ͷ
	bool m_isClose;
public:
	// ����ͷ
	VideoCapture m_capture;
	HDC m_capHdc;
	CRect m_capRect;
	IplImage* faceImage;
	Mat faceImgShow;
	Mat captureImg;//registerͼƬ
	Mat frame;//����ͷͼƬ
};

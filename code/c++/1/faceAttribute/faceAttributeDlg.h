
// faceAttributeDlg.h : ͷ�ļ�
//

#pragma once
#include "register.h"
#include "atltypes.h"
#include <opencv2/opencv.hpp>
#include "faceDetect.h"
#include "faceRecognition.h"
#include "RegisterBox.h"
// CfaceAttributeDlg �Ի���
class CfaceAttributeDlg : public CDialogEx
{
// ����
public:
	CfaceAttributeDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_FACEATTRIBUTE_DIALOG };

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
	// ��ʾͼ�񴰿�DC
	HDC m_picHdc;
	// ��ʾ����ͷͼƬ����rect
	CRect m_picRect;
	// ע��DC
	HDC m_recordHdc;
	CRect m_recordRect;

	// ����ͷ�Ƿ���
	bool m_isStart;
	// �Ƿ�����
	bool m_isRecord;

	// �Ƿ�ʶ��
	bool m_isRecognition;

	// ����ͷ
	VideoCapture m_capture;
	// ע��ͼƬ
	Mat m_captureImg;

	//�������ָ��
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

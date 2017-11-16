#pragma once
#include "atltypes.h"


// CCompare �Ի���

class CCompare : public CDialogEx
{
	DECLARE_DYNAMIC(CCompare)

public:
	CCompare(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~CCompare();

// �Ի�������
	enum { IDD = IDD_COMPARE_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

	DECLARE_MESSAGE_MAP()
public:
	HDC m_picHdc_1;
	HDC m_picHdc_2;
	CRect m_picRect_1;
	CRect m_picRect_2;
	CString m_similarity;
	afx_msg void OnBnClickedBtn1();
	CString m_file1;
	CString m_file2;
	cv::Mat m_img1;
	cv::Mat m_img2;
	afx_msg void OnBnClickedBtn2();
	void showImage(Mat& im, UINT ID);
	afx_msg void OnBnClickedBtnCampare();
};

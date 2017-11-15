
// faceAnalysisDlg.h : ͷ�ļ�
//

#pragma once
#include "register.h"
#include "processThread.h"

// CfaceAnalysisDlg �Ի���
class CfaceAnalysisDlg : public CDialogEx
{

private:
	ProcessThread* faceDetectThread;
	

// ����
public:
	CfaceAnalysisDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_FACEANALYSIS_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��

	Mat faceImage;
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
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedBtnCamera();
};

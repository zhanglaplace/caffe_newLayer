
// faceAnalysisDlg.h : 头文件
//

#pragma once
#include "register.h"
#include "processThread.h"

// CfaceAnalysisDlg 对话框
class CfaceAnalysisDlg : public CDialogEx
{

private:
	ProcessThread* faceDetectThread;
	

// 构造
public:
	CfaceAnalysisDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_FACEANALYSIS_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持

	Mat faceImage;
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
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedBtnCamera();
};

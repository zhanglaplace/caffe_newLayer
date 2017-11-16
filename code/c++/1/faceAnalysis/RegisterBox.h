#pragma once
#include "afxwin.h"


// CRegisterBox 对话框

class CRegisterBox : public CDialog
{
	DECLARE_DYNAMIC(CRegisterBox)

public:
	CRegisterBox(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~CRegisterBox();
// 对话框数据
	enum { IDD = IDD_REGISTER_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()


public:
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedCancel();
	CString m_personName;
};

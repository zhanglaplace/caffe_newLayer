#pragma once
#include "afxwin.h"


// CRegisterBox �Ի���

class CRegisterBox : public CDialog
{
	DECLARE_DYNAMIC(CRegisterBox)

public:
	CRegisterBox(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~CRegisterBox();
// �Ի�������
	enum { IDD = IDD_REGISTER_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

	DECLARE_MESSAGE_MAP()


public:
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedCancel();
	CString m_personName;
};

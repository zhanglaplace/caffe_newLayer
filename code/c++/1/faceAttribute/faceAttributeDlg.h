
// faceAttributeDlg.h : ͷ�ļ�
//

#pragma once
#include "register.h"

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
};
